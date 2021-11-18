import cv2
from importlib import import_module
import os
import os.path as osp
import time
import math
import random
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

from detect import detect
from deteval import *
import json

import wandb
from glob import glob


def fix_seed(random_seed=0):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def seed_worker(worker_id):
    """
    fix seed for multi-processing in the process of loading data.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/'))
    parser.add_argument('--json_dir', type=str,
                        default='ufo/train.json')
    parser.add_argument('--valid_json_dir', type=str,
                        default='ufo/train.json')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))
    parser.add_argument('--dataset_type', type=str, default='PolygonDatasetExceptCrop')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--use_poly', type=bool, default=False)
    parser.add_argument('--use_val', type=bool, default=False)
    parser.add_argument('--use_fp16', type=bool, default=False)

    parser.add_argument('--pretrain_last', type=bool, default=False)
    parser.add_argument('--pretrained_last_dir', type=str, default='/opt/ml/code/trained_models/latest.pth')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_validating(model, data_dir, valid_json_dir, batch_size, data_loader=None, num_batches=None, epoch=None,input_size = 1024):
    """
    key값에서 Loss를 출력하냐 metric을 출력하냐에 따라 필요한 key값이 다른데 일단 다 넣어 주고 metric 출력은 주석 처리해줬습니다.
    Loss의 경우 필요한 key 값은 model, data_loader, num_batches, epoch 
    metric의 경우 필요한 model, data_dir, valid_json_dir, batch_size 입니다.
    """

    model.eval()
    ###############
    ## num1 LOSS ##
    ###############

    total_loss = 0
    epoch_start = time.time()
    with tqdm(total=num_batches) as pbar:
        for img, gt_score_map, gt_geo_map, roi_mask in data_loader:
            pbar.set_description('[Validation Epoch {}]'.format(epoch + 1))

            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            
            total_loss += loss.item()

            pbar.update(1)
            val_dict = {
                'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                'IoU loss': extra_info['iou_loss']
            }
            pbar.set_postfix(val_dict)
    print('Validation Mean loss: {:.4f} | Elapsed time: {}'.format(
    total_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

    return total_loss/num_batches, extra_info['cls_loss'], extra_info['angle_loss'], extra_info['iou_loss']

    #################
    ## NUM2 METRIC ##
    #################

#     with open(osp.join(data_dir,valid_json_dir), "r") as json_file:
#         valid_json = json.load(json_file)
#     valid_json_imgs = valid_json['images'].keys()

#     image_fnames, by_sample_bboxes = [], []
#     # total_loss = 0

#     images = []
#     for image_fpath in tqdm(valid_json_imgs): # 이미지 파일 경로 불러오고
#         image_fnames.append(image_fpath)

#         images.append(cv2.imread("/opt/ml/input/data/ICDAR17_OW/images/"+image_fpath)[:, :, ::-1]) # 이미지 하나 로드 채널 변경
#         if len(images) == batch_size: # 배치 사이즈 축적
#             by_sample_bboxes.extend(detect(model, images, input_size)) #inference하고 전체 저장소에대가 저장
#             images = [] # 초기화
    
#     if len(images): # 남아 있다면 마지막 예측
#         by_sample_bboxes.extend(detect(model, images, input_size))

# #     ufo_result = dict(images=dict()) # ufo 저장할 딕트 생성
#     pred_bboxes_dict = dict()
#     for image_fname, bboxes in zip(image_fnames, by_sample_bboxes): # 이미지_id, 예측된 박스
#         pred_bboxes_dict[image_fname] = [bbox.tolist() for bbox in bboxes]

#     gt_bboxes_dict = dict()
#     transcriptions_dict = dict()
#     eval_hparams = dict(AREA_RECALL_CONSTRAINT=0.8, AREA_PRECISION_CONSTRAINT=0.4,
#     EV_PARAM_IND_CENTER_DIFF_THR=1, MTYPE_OO_O=1.0, MTYPE_OM_O=0.8, MTYPE_OM_M=1.0)
    
#     for img_key in valid_json_imgs:
#         gt_bboxes_dict[img_key]=[]
#         transcriptions_dict[img_key] = []
#         for word_key in valid_json['images'][img_key]['words']:
#             gt_bboxes_dict[img_key].append(valid_json['images'][img_key]['words'][word_key]['points'])
#             transcriptions_dict[img_key].append(valid_json['images'][img_key]['words'][word_key]['transcription'])
    
#     resDict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict, eval_hparams)
#     print(f"precision: {resDict['total']['precision']}, recall: {resDict['total']['recall']}, f1-score(hmean): {resDict['total']['hmean']}")
#     return resDict['total']['precision'], resDict['total']['recall'], resDict['total']['hmean']



def do_training(seed,data_dir, json_dir, model_dir, dataset_type, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, pretrain_last, pretrained_last_dir, use_poly=True, use_val=False, valid_json_dir=None, use_fp16=False):


    # code for a reproducibility
    fix_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)


    # import dataset from dataset.py and build a loader
    dataset_module = getattr(import_module("dataset"), dataset_type)
    data_dir = glob(osp.join(data_dir,'*'))
    dataset = dataset_module(data_dir, split='train', image_size=image_size, target_size=input_size, use_poly=use_poly)
    dataset = EASTDataset(dataset) 
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
                            dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers, 
                            worker_init_fn=seed_worker, 
                            generator=g)


    # when use a validation set
    if use_val:
        valid_dataset = dataset_module(data_dir, split='train', image_size=image_size, crop_size=input_size) # 나중에 error handling 해줘야할 듯.
        valid_dataset = EASTDataset(valid_dataset) 
        valid_loader = DataLoader(
                            valid_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers, 
                            worker_init_fn=seed_worker, 
                            generator=g)
        valid_num_batches = math.ceil(len(valid_dataset) / batch_size)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not pretrain_last: # 이미지넷 기학습 가중치를 불러오는 경우
        model = EAST()
    else: # 직접 학습시킨 결과물의 가중치를 불러오는 경우
        model = EAST(pretrained=False)
        model.load_state_dict(torch.load(pretrained_last_dir))
    
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    
    wandb.watch(model)

    if use_fp16:
        print("Mixed precision is applied")
        scaler = GradScaler()

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                optimizer.zero_grad()

                if use_fp16:
                    with autocast():
                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    loss.backward()
                    optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
                wandb.log(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if use_val: 
            val_mean_loss, val_cls_loss, val_angle_loss, val_iou_loss = do_validating(model, data_dir, valid_json_dir, batch_size, valid_loader, valid_num_batches, epoch)

        wandb.log(
            {"Epoch Mean loss": epoch_loss / num_batches}
        )

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    wandb.init(project="ocr", name="quad_polygon_data_refined")
    wandb.config.update(args)
    do_training(**args.__dict__)
    wandb.run.finish()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
