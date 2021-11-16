import json
import pandas as pd
import os.path as osp
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedGroupKFold

if __name__=="__main__":
    parser = ArgumentParser(description='')
    parser.add_argument('--json_path', type = str, 
                        help='json file path',default='/opt/ml/input/data/ICDAR17_Korean/ufo/train.json')
    parser.add_argument('--split_num', type = str, 
                        help='fold split number',default=2)
    parser.add_argument('--seed', type = str, 
                        help='fold split seed',default=42)
    parser.add_argument('--json_save_folder', type = str, 
                        help='folder_directory to save new json file', default='/opt/ml/input/data/ICDAR17_Korean/ufo')
    args = parser.parse_args() 


    with open(args.json_path) as f:
        train_json = json.load(f)

    train_df = pd.DataFrame(columns=['img', 'img_h', 'img_w', 'tags', 'usability', 'public', 'commercial',
       'type', 'holder', 'points', 'transcription', 'language', 'illegibility',
       'orientation', 'word_tags'])

    print("making dataframe...")
    for img_name in tqdm(train_json['images'].keys()):
        other_info = pd.concat([pd.Series([img_name], name = 'img'), 
            pd.json_normalize(train_json['images'][img_name])[['img_h','img_w','tags']],
            pd.json_normalize(train_json['images'][img_name]['license_tag'])], axis = 1)
        words_info = pd.json_normalize(train_json['images'][img_name]['words'].values())
        img_concat_info = pd.concat([pd.concat([other_info]*words_info.shape[0]).reset_index(drop=True),words_info], axis = 1)
        train_df = train_df.append(img_concat_info).reset_index(drop=True)
    
    train_df['language'] = train_df['language'].apply(lambda x: x[0])
    
    skgf = StratifiedGroupKFold(n_splits = args.split_num, random_state = args.seed, shuffle = True)
    folds = skgf.split(train_df.index, train_df['language'], train_df['img'])

    for fold, (trn_idx, val_idx) in enumerate(folds):
        print(f"{fold} spliting...")
        trn_img = train_df.loc[trn_idx,'img'].unique().tolist()
        val_img = train_df.loc[val_idx,'img'].unique().tolist()

        for img_ids, tr_val in zip([val_img, trn_img], ['train', 'valid']):
            with open(args.json_path) as f:
                train_origin_json = json.load(f)
            for k in img_ids:
                train_origin_json['images'].pop(k, None)
            with open(osp.join(args.json_save_folder,"split_"+tr_val+'.json'), 'w') as fp:
                json.dump(train_origin_json, fp)    
        if fold == 0: break;

