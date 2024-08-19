import os
import clip
import json
import torch
import random
import pickle
import logging
import argparse
import transformers
import numpy as np
import torch.nn as nn
from collections import OrderedDict

from tqdm import tqdm
from dataloader import get_it_loader
from evaluate import eval_epoch
# from sentence_transformers import SentenceTransformer
from train import merge_dict


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.distributed.init_process_group(backend="nccl")

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger


def set_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    set_seed(args)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(filename=args.output_dir+'/log.txt')

    clip_ckpt = args.clip_ckpt

    langs = args.langs.split(',')
    clip_model, preprocess = clip.load(clip_ckpt, device='cuda', acquirer=args.acquirer, d_acquirer_hidden=args.acquirer_hidden, m_acquirer=args.m_acquirer, langs=langs, skip=args.skip)
    src_toker = clip.tokenize
    clip_model.eval()

    trg_toker = transformers.AutoTokenizer.from_pretrained('pretrained_model/bert-base-multilingual-cased')

    assert args.init_model is not None or (args.is_clip or args.is_mclip or args.sentence_transformer)
    
    if args.acquirer_ckpt:
        logger.info(f'Loading acquirer weight from {args.acquirer_ckpt}')
        acquirer_state_dict = torch.load(args.acquirer_ckpt)
        acquirer_keys = ['acquirer', 'multilingual_embedding', 'multilingual_embedding_linear']
        acquirer_state_dict = OrderedDict({
            k:v for k,v in acquirer_state_dict.items() if any(n in k for n in acquirer_keys)
        })
        clip_model.load_state_dict(acquirer_state_dict, strict=False)

    if args.embedding_ckpt:
        logger.info(f'Loading embedding weight from {args.embedding_ckpt}')
        embedding_state_dict = torch.load(args.embedding_ckpt)
        embedding_keys = ['multilingual_embedding', 'multilingual_embedding_linear']
        embedding_state_dict = OrderedDict({
            k:v for k,v in embedding_state_dict.items() if any(n in k for n in embedding_keys)
        })
        clip_model.load_state_dict(embedding_state_dict, strict=False)

    test_loaders = []

    if args.img_type == 'm30k+coco':
        path_dict_m30k = json.load(open('./dataset/Multi30k/multi30k_4lang_path.json'))
        path_dict_coco = json.load(open('./dataset/MSCOCO/COCO_en_path.json'))
        path_dict = merge_dict(path_dict_m30k, path_dict_coco)
    elif args.img_type == 'm30k':
        path_dict = json.load(open('./dataset/Multi30k/multi30k_4lang_path.json')) 

    anno_files = path_dict[args.eval_split]['anno_file']
    image_dirs = path_dict[args.eval_split]['feature']
    name_files = path_dict[args.eval_split]['name_file']
    langs = path_dict[args.eval_split]['langs']

    print(langs)

    split_test_loaders = []
    split_test_lang = []
    none_split_lang = []

    for anno, image_dir, name_file, lang in zip(anno_files, image_dirs, name_files, langs):

        if isinstance(name_file, list):
            temp_loader = []
            temp_split = []
            for i, name in enumerate(name_file):
                
                _, loader_split = get_it_loader(args, name, anno, image_dir, preprocess, src_toker, is_train=False, st=args.sentence_transformer, return_type='dict')
                temp_loader.append(loader_split)
            split_test_loaders.append(temp_loader)
            split_test_lang.append(lang)
        else:
            _, loader = get_it_loader(args, name_file, anno, image_dir, preprocess, src_toker, is_train=False, st=args.sentence_transformer, return_type='dict')
            test_loaders.append(loader)
            none_split_lang.append(lang)
    
    for split_test_loader, lang in zip(split_test_loaders, split_test_lang):
        if 'ja' not in lang:
            continue
        metrics = eval_epoch(clip_model, split_test_loader, 'cuda', 1, [f'{lang}_split_{i}' for i in range(len(split_test_loader))], trg_toker, logger, is_clip=args.is_clip, return_all_metrics=True, sentence_transformer=args.sentence_transformer,
        acquirer=args.acquirer, new_embed=args.new_embed, save_matrix=True, output_root=os.path.join(args.output_dir, 'sim_matrix')
        )
        # logger.info(f' Lang: {lang} mR1: {mR1}')

    metrics = eval_epoch(clip_model, test_loaders, 'cuda', 1, none_split_lang, trg_toker, logger, is_clip=args.is_clip, return_all_metrics=True, sentence_transformer=args.sentence_transformer,
        acquirer=args.acquirer, new_embed=args.new_embed, save_matrix=True, output_root=os.path.join(args.output_dir, 'sim_matrix')
        )

    # logger.info(f'mR1: {mR1}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--init_model', default=None, type=str)
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--itm', action='store_true')
    parser.add_argument('--itm_nce', action='store_true')
    parser.add_argument('--kd', default=True, type=int)
    parser.add_argument('--optim', choices=['Adam', 'BertAdam'], type=str, default='Adam')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--img_type', type=str, default='m30k+coco')
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--is_clip', action='store_true')
    parser.add_argument('--is_mclip', action='store_true')
    parser.add_argument('--clip_ckpt', type=str, default='pretrained_model/clip/ViT-B-32.pt')
    parser.add_argument('--sentence_transformer', action='store_true')

    parser.add_argument('--acquirer', action='store_true')
    parser.add_argument('--new_embed', action='store_true')
    parser.add_argument('--m_acquirer', action='store_true')
    parser.add_argument('--langs', type=str, default='de,fr,cs,ja,zh,it,es,ru,pl,tr,ko')
    parser.add_argument('--acquirer_ckpt', type=str, default=None)
    parser.add_argument('--embedding_ckpt', type=str, default=None)
    parser.add_argument('--acquirer_hidden', type=int, default=256)
    parser.add_argument('--mbert_acquirer', action='store_true')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--acquirer_config', type=str, default='./expr/mbert_acquirer/acquirer_config.json')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--path_file', type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/eval_config.json', 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, indent=1, ensure_ascii=False)
    main(args)
