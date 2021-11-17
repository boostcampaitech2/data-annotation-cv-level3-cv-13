import glob
import json
import os.path as osp
from PIL import Image, ImageOps
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd



# license tag 고정
# 학습시에는 상관없기에 img_width, height, orientation, tag 값 임의로 넣어주었습니다. 
# 참고 language: ["Latin", "None", "Korean", "Arabic", "Symbols", "Chinese", "Bangla", 'Japanese', "Mixed"}]

if __name__=="__main__":
    parser = ArgumentParser(description='')
    parser.add_argument('--gt_text_file_path', type = str, 
                        help='Ground Truth text file directory',default='/opt/ml/input/data/ICDAR17_OW/text_gt')
    parser.add_argument('--img_file_path', type = str, 
                        help='image files directory',default='/opt/ml/input/data/ICDAR17_OW/images')
    parser.add_argument('--rm_lan', nargs='+',
                        help='language you want to remove',default=[]) # ex) --rm_lan Arabic Symbols Chinese Bangla Japanese Mixed 
    parser.add_argument('--json_dir', type = str, 
                        help='Path where json file is saved',default="/opt/ml/input/data/ICDAR17_OW")
    parser.add_argument('--valid_json_dir', type = str, 
                    help='it is used for eliminating validation image in ICDAR17 json',default=None)

    args = parser.parse_args() 


    icdar_json = {'images':{}}
    license_tag = {'license_tag': {'usability': True,
   'public': True,
   'commercial': True,
   'type': 'CC-BY-SA',
   'holder': None}}


    for txt_file in glob.glob(osp.join(args.gt_text_file_path,"*.txt")):
        with open(txt_file) as f:
            contents = f.readlines()
        img_id = txt_file.split('_')[-2]+"_"+txt_file.split('_')[-1].split(".")[0] + '.jpg'
        img_json = {img_id:{}}
        img_json[img_id].update({'img_h':0})
        img_json[img_id].update({'img_w':0})

        words = {'words':{}}
        for i, words_info in enumerate(contents):
            word = {str(i):{}}
            points_list = list(map(int,words_info.split(',')[:8]))
            points = [[x,y] for x, y in zip(points_list[::2],points_list[1::2])]
            language = words_info.split(',')[8]
            ocr_word = ''.join(words_info.split(',')[9:]).strip()
            illegibility = True if ocr_word == '###' else False
            word[str(i)].update({'points':points})
            word[str(i)].update({'transcription':ocr_word})
            word[str(i)].update({'language':[language]})
            word[str(i)].update({'illegibility':illegibility})
            word[str(i)].update({'orientation':'Horizontal'})
            word[str(i)].update({'word_tags':None})
            words['words'].update(word)
        img_json[img_id].update(words)
        img_json[img_id].update({'tags':None})
        img_json[img_id].update(license_tag)
        icdar_json['images'].update(img_json)
    print(f"전체 이미지수: {len(icdar_json['images']):,}")

    # ICDAR 이미지에 jpg가 아닌 gif, png로 저장된 이미지를 jpg로 형 변환
    # 이미지 처리하고 싶지 않다면 주석처리 해주세요 ~
    for img_format in ["*.gif", "*.png"]:
        for img_file in glob.glob(osp.join(args.img_file_path, img_format)):
            img = Image.open(img_file)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB") # https://codedragon.tistory.com/6118
            img.save(img_file.split('.')[0]+'.jpg')        

    # 지우고자 하는 언어가 있다면 해당 언어를 포함하고 있는 이미지 제거
    if args.rm_lan:
        print(f"지우고자 하는 언어: {args.rm_lan}")
        rm_img_list = []
        for img_name in tqdm(icdar_json['images'].keys()):
            lan_list = []
            for word_key in icdar_json['images'][img_name]['words']:
                lan_list.extend(icdar_json['images'][img_name]['words'][word_key]['language'])
            if set(lan_list).intersection(set(args.rm_lan)):
                rm_img_list.append(img_name)
        for k in rm_img_list:
            icdar_json['images'].pop(k, None)
        print(f"제거되고 남은 이미지수: {len(icdar_json['images']):,}")
      
    
    if args.valid_json_dir:
        with open(args.valid_json_dir, "r") as json_file:
            validation_json = json.load(json_file)
        for k in validation_json['images'].keys():
            icdar_json['images'].pop(k, None)
        print(f"Validation image가 제거되고 남은 이미지수: {len(icdar_json['images']):,}") 

    with open(osp.join(args.json_dir, 'icdar17.json'), 'w') as fp:
        json.dump(icdar_json, fp)
