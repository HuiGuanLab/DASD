import os
import json
from torch.utils.data import ConcatDataset, DataLoader
from dataloader import get_it_loader, get_mit_loader, ImageTextDataset
from data.loader import MetaLoader , IterLoader

def get_cc(args, preprocess, is_train, langs=['en','de','fr','cs','zh','ja'], top=-1, train_en=False):
    root_path = 'dataset/ConceptualCaption/annotation/uc2_6langs/'
    image_dir = 'dataset/ConceptualCaption/images/'
    names = 'dataset/ConceptualCaption/annotation/uc2_6langs/shuffled_names.npy'
    data_list = []
    for lang in langs:
        anno_path = os.path.join(root_path, 'jsons', f'ref_captions_{lang}.json')
        data_list.append(anno_path)
    
    return get_mit_loader(args, names, data_list, langs, image_dir, preprocess, is_train, top=top, train_en=train_en)

##gai
def get_cc_iter(args, preprocess, is_train, langs, top=-1, train_en=False,stage='NLT'):
    file_path=args.data_path
    root_path =os.path.join(file_path,'ConceptualCaption/annotation/uc2_6langs/')
    image_dir=os.path.join(file_path,'ConceptualCaption/images/')
    names = os.path.join(file_path,'ConceptualCaption/annotation/uc2_6langs/shuffled_names.npy')
    en2zh_path = os.path.join(file_path,'ConceptualCaption/trans_data/ja2en_json_file.json')
    data_list = []
    loaders = {}
    for lang in langs:
        if lang == 'en' and (not train_en):
            continue
        anno_path = os.path.join(root_path, 'jsons', f'new_ref_captions_{lang}.json')
        # data_list.append(anno_path)
        print(f'Constructing {lang} loader')
        # loaders.append(
        #     (lang, (get_mit_loader(args, names, [anno_path], ['en', lang], image_dir, preprocess, is_train, top=top)[-1], 1))
        # )
        loaders[lang] = (get_mit_loader(args, names, [os.path.join(root_path, 'jsons', f'new_ref_captions_en.json'), anno_path], ['en', lang], image_dir, preprocess, is_train, top=top, stage=stage, en2zh_path=en2zh_path)[-1], 1)
    return MetaLoader(loaders)

def get_general_iter(args, path_file, preprocess, top=-1, train_en=False):
    path_dict = json.load(open(path_file))
    loaders = {}
    annos_files = path_dict['train']['anno_file']
    image_dirs = path_dict['train']['feature']
    name_files = path_dict['train']['name_file']
    langs = path_dict['train']['langs']
    lang2anno = {}
    lang2imagedir = {}
    lang2names = {}
    for i, lang in enumerate(langs):
        lang2anno[lang] = annos_files[i]
        lang2imagedir[lang] = image_dirs[i]
        lang2names[lang] = name_files[i]
    
    src_lang = 'en'
    trg_langs = [lang for lang in langs if lang != src_lang]
    src_anno, src_imagedir, src_names = lang2anno[src_lang], lang2imagedir[src_lang], lang2names[src_lang]
    for i, lang in enumerate(trg_langs):
        trg_anno = lang2anno[lang]
        image_dir = lang2imagedir[lang]
        _, mit_loader = get_mit_loader(args, src_names, [src_anno, trg_anno], ['en', lang], image_dir, preprocess, False, top=top)
        loaders[lang] = (mit_loader, 1)

    return MetaLoader(loaders)

def get_general_eval(args, path_file, preprocess, toker, split):
    eval_loaders, eval_langs = [], []
    path_dict = json.load(open(path_file))
    anno_files = path_dict[split]['anno_file']
    image_dirs = path_dict[split]['feature']
    name_files = path_dict[split]['name_file']
    langs = path_dict[split]['langs']
    for anno, image_dir, name_file, lang in zip(anno_files, image_dirs, name_files, langs):
        _, loader = get_it_loader(args, name_file, anno, image_dir, preprocess, toker, is_train=False)
        eval_loaders.append(loader)
        eval_langs.append(lang)
    return eval_loaders, eval_langs

def get_m30k(args, preprocess, split='train', langs=['en','de','fr','cs']):
    file_path=args.data_path

    json_path = os.path.join(file_path,'Multi30k/multi30k_4lang_path.json')

    path_dict = json.load(open(json_path))
    anno_files = path_dict[split]['anno_file']
    features = path_dict[split]['feature']
    name_files = path_dict[split]['name_file']
    all_langs = path_dict[split]['langs']
    datasets = []
    for anno, feat, name, lang in zip(anno_files, features, name_files, all_langs):
        if lang not in langs:
            continue
        dataset = ImageTextDataset(
            name, anno, feat, preprocess, toker=None, sentence_transformer=False
        )
        datasets.append(dataset)
    pair_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        dataset=pair_dataset,
        batch_size=args.batch_size,
        shuffle=(split=='train'),
        num_workers=args.n_workers,
        drop_last=(split=='train')
    )
    return dataset, dataloader

def get_m30k_iter(args, preprocess, split='train', langs=['en','de','fr','cs']):
    file_path=args.data_path
    json_path = os.path.join(file_path,'Multi30k/multi30k_4lang_path.json')
    path_dict = json.load(open(json_path))
    anno_files = path_dict[split]['anno_file']
    features = path_dict[split]['feature']
    name_files = path_dict[split]['name_file']
    all_langs = path_dict[split]['langs']
    loaders = {}
    for anno, feat, name, lang in zip(anno_files, features, name_files, all_langs):
        if lang not in langs:
            continue
        dataset = ImageTextDataset(
            name, anno, feat, preprocess, toker=None, sentence_transformer=False
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=(split=='train'),
            num_workers=args.n_workers,
            drop_last=(split=='train')
        )
        loaders[lang] = (dataloader, 1)
    return MetaLoader(loaders)

def get_coco_iter(args, preprocess, split='train', langs=['en','ja','zh']):
    file_path=args.data_path
    json_path = os.path.join(file_path,'MSCOCO_translate/COCO_all_split0_path.json')    
    path_dict = json.load(open(json_path))
    anno_files = path_dict[split]['anno_file']
    features = path_dict[split]['feature']
    name_files = path_dict[split]['name_file']
    all_langs = path_dict[split]['langs']
    loaders = {}
    for anno, feat, name, lang in zip(anno_files, features, name_files, all_langs):
        if lang not in langs:
            continue
        dataset = ImageTextDataset(
            name, anno, feat, preprocess, toker=None, sentence_transformer=False,stage=args.stage
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=(split=='train'),
            num_workers=args.n_workers,
            drop_last=(split=='train')
        )
        loaders[lang] = (dataloader, 1)
    return MetaLoader(loaders)

def get_coco(args, preprocess, split='train', langs=['en','ja','zh']):
    file_path=args.data_path
    json_path = os.path.join(file_path,'MSCOCO_translate/COCO_all_split0_path.json')
    path_dict = json.load(open(json_path))
    anno_files = path_dict[split]['anno_file']
    features = path_dict[split]['feature']
    name_files = path_dict[split]['name_file']
    all_langs = path_dict[split]['langs']
    datasets = []
    for anno, feat, name, lang in zip(anno_files, features, name_files, all_langs):
        if lang not in langs:
            continue
        dataset = ImageTextDataset(
            name, anno, feat, preprocess, toker=None, sentence_transformer=False
        )
        datasets.append(dataset)
    pair_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        dataset=pair_dataset,
        batch_size=args.batch_size,
        shuffle=(split=='train'),
        num_workers=args.n_workers,
        drop_last=(split=='train')
    )
    return dataset, dataloader
