import os.path as osp
import math
import json
from PIL import Image, ImageOps
from imageio import imread

import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Polygon

from augmentation import *

def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, start_w + length, start_h + length,
                  start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def resize_img(img, vertices, size):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    new_vertices = vertices * ratio
    return img, new_vertices


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices


def generate_roi_mask(image, vertices, labels):
    """
    illegibility = True 인 곳에만 zero-masking
    """
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels


def split_polygon_to_quadril(word_info):
    pts = word_info["points"]
    num_pts = len(pts)
    if num_pts % 2 != 0 or num_pts <= 2:
        assert 'Number of points should be even and bigger than 2.'

    result = []
    if word_info["orientation"] == "Horizontal":
        for i in range(0, num_pts//2 - 1):
            r = []
            r.append(pts[i])
            r.append(pts[i+1])
            r.append(pts[num_pts-i-2])
            r.append(pts[num_pts-i-1])
            result.append(np.array(r).flatten())
    else:
        for i in range(1, num_pts//2):
            r = []
            r.append(pts[(num_pts-i+1) % num_pts])
            r.append(pts[i])
            r.append(pts[i+1])
            r.append(pts[(num_pts-i+1) % num_pts - 1])
            result.append(np.array(r).flatten())

    return result

def transform_to_quad(points):
    """
    - Args
        points: 8개 초과하는 임의의 polygon의 좌표들 (1d)
    """
    divider = len(points)//2
    upside = points[:divider]
    downside = points[divider:]

    x1, y1 = min(upside[::2]), min(upside[1::2])
    x2, y2 = max(upside[::2]), min(upside[1::2])
    x3, y3 = max(downside[::2]), max(downside[1::2]) 
    x4, y4 = min(downside[::2]), max(downside[1::2])

    return np.array([x1, y1, x2, y2, x3, y3, x4, y4])


def set_target_size(img, vertices, size):
    h, w = img.height, img.width
    target_h, target_w = size, size
    img = img.resize((target_w, target_h), Image.BILINEAR)
    ratio_w = target_w / w
    ratio_h = target_h / h
    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h
    return img, new_vertices

    
class SceneTextDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=1024, target_size=512, color_jitter=True,
                 normalize=True, use_poly = True):
        with open(osp.join(root_dir, 'ufo/{}.json'.format(split)), 'r') as f:
            anno = json.load(f)

        self.anno = anno
        self.image_fnames = sorted(anno['images'].keys())
        self.image_dir = osp.join(root_dir, 'images')
        self.use_poly = use_poly
        self.image_size, self.crop_size = image_size, target_size
        self.color_jitter, self.normalize = color_jitter, normalize

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self.image_dir, image_fname)

        vertices, labels = [], []
        
        for word_info in self.anno['images'][image_fname]['words'].values():
            if self.use_poly:
                tmp = split_polygon_to_quadril(word_info)
                vertices.extend(tmp)
                labels.extend([int(not word_info['illegibility'])] * len(tmp))

            else:
                flattened_vertices = np.array(word_info['points']).flatten()
                if len(flattened_vertices) > 8:
                    flattened_vertices = transform_to_quad(flattened_vertices)
                elif len(flattened_vertices) < 8:
                    # annotation information이 완전하지 않을 경우 버린다.
                    continue
                vertices.append(flattened_vertices) # flatten을 해주면 4,2 -> 4*2
                labels.append(int(not word_info['illegibility'])) # 유의하면 1 아니면 0
        
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64) # 이 부분에서 무조건 inhomogeneous shape error가 날 수밖에 없음. 애초에 np array는 이걸 안받아.
        vertices, labels = filter_vertices(vertices, labels, ignore_under=10, drop_under=1)

        image = Image.open(image_fpath)
        image = ImageOps.exif_transpose(image) # If the image rotates automatically, it changes it to its original state.

        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, self.crop_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        funcs = []
        if self.color_jitter:
            funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = A.Compose(funcs)

        image = transform(image=image)['image']
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask


class TransformedSceneTextDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        with open(osp.join(root_dir, 'ufo/{}.json'.format(split)), 'r') as f:
            anno = json.load(f)

        self.anno = anno
        self.image_fnames = sorted(anno['images'].keys())
        self.image_dir = osp.join(root_dir, 'images')

        self.transform = transform

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self.image_dir, image_fname)

        vertices, labels = [], []
        for word_info in self.anno['images'][image_fname]['words'].values():
            vertices.append(np.array(word_info['points']))
            labels.append(int(not word_info['illegibility']))
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        image = imread(image_fpath, pilmode='RGB')
        if self.transform != None:
            if isinstance(self.transform, ComposedTransformation):
                transformed = self.transform(image=image, word_bboxes=vertices)
                image = transformed['image']
                vertices = transformed['word_bboxes']
            else:
                pass

        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask


class PolygonDatasetExceptCrop(Dataset):
    '''
    기존 SceneTextDatset에서 crop 대신 resize를 수행합니다.

    Args:
        root_dir (str): json파일과 이미지파일의 상위 디렉토리
        split (str): json파일명
        image_size (int): 이미지 resize시 평균 크기
        target_size (int): 이미지 최종 크기
        color_jitter (bool): color_jitter 적용 유무
        normalize (bool): normalize 적용 유무

    Attributes:
        anno (dict): json파일 load 내용
        image_fnames (list): 이미지 파일명 리스트 
        image_dir (str): 이미지 폴더 경로
        image_size (int): 이미지 resize시 평균 크기
        target_size (int): 이미지 최종 크기
        color_jitter (bool): color_jitter 적용 유무
        normalize (bool): normalize 적용 유무
    '''
    def __init__(self, root_dir, split='train', image_size=1024, target_size=1024, color_jitter=True,
                 normalize=True, use_poly = True):

        self.annos = []
        self.image_dir = []
        for dir in root_dir:
            self.image_dir.append(osp.join(dir, 'images'))
            with open(osp.join(dir, 'ufo/{}.json'.format(split)), 'r') as f:
                self.annos.append(json.load(f))
        self.image_fnames = []
        # 받아온 이미지 리스트를 여기서 1개로 합쳐줍니다.
        for anno in self.annos:
            self.image_fnames.extend(sorted(anno['images'].keys()))
        self.use_poly = use_poly

        self.image_size, self.target_size = image_size, target_size
        self.color_jitter, self.normalize = color_jitter, normalize

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        vertices, labels = [], []
        for i, anno in enumerate(self.annos):
            if image_fname in anno['images'].keys():
                image_fpath = osp.join(self.image_dir[i],image_fname)
                for word_info in anno['images'][image_fname]['words'].values():
                    if self.use_poly:
                        tmp = split_polygon_to_quadril(word_info)
                        vertices.extend(tmp)
                        labels.extend([int(not word_info['illegibility'])] * len(tmp))

                    else:
                        flattened_vertices = np.array(word_info['points']).flatten()
                        if len(flattened_vertices) > 8:
                            flattened_vertices = transform_to_quad(flattened_vertices)
                        elif len(flattened_vertices) < 8:
                            # annotation information이 완전하지 않을 경우 버린다.
                            continue
                        vertices.append(flattened_vertices) # flatten을 해주면 4,2 -> 4*2
                        labels.append(int(not word_info['illegibility'])) # 유의하면 1 아니면 0
                vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)
                vertices, labels = filter_vertices(vertices, labels, ignore_under=10, drop_under=1)

        image = Image.open(image_fpath)
        image = ImageOps.exif_transpose(image) # If the image rotates automatically, it changes it to its original state.
        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        image, vertices = set_target_size(image, vertices, self.target_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        funcs = []
        if self.color_jitter:
            funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = A.Compose(funcs)

        image = transform(image=image)['image']
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask