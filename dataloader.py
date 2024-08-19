import os
import json
import copy
import re
import torch
import numpy as np
import torch as Tensor
from data.loader import MetaLoader , IterLoader
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random
from transformers import BertTokenizer

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class TxtReader(object):
    def __init__(self, txt_path, max_sents=-1):
        self.lines = open(txt_path, 'r', encoding='utf-8').readlines()
        self.lines = [line.strip() for line in self.lines]
        if max_sents != -1:
            self.lines = self.lines[:max_sents]
    
    def __getitem__(self, idx):
        return self.lines[idx]
    
    def __len__(self):
        return len(self.lines)

class TextDataset(Dataset):
    def __init__(self, src_reader, src_toker, max_len=50):
        super().__init__()
        self.src = src_reader
        self.src_toker = src_toker
        self.max_len = max_len

    def __getitem__(self, idx):
        src_sent = self.src[idx]
        src_dict = self.src_toker(src_sent, truncation=True, max_length=self.max_len, padding='max_length')
        src_ids = src_dict['input_ids']
        src_mask = src_dict['attention_mask']
        
        # import pdb;pdb.set_trace()

        return Tensor(src_ids), Tensor(src_mask)

        # return {
        #     'src_ids': src_ids,
        #     'src_mask': src_mask,
        #     'trg_ids': trg_ids,
        #     'trg_mask': trg_mask
        # }

    def __len__(self):
        return len(self.src)

class PairTextDataset(Dataset):
    def __init__(self, src_reader, trg_reader, src_toker, trg_toker, max_len=50):
        super().__init__()
        self.src = src_reader
        self.trg = trg_reader
        self.src_toker = src_toker
        self.trg_toker = trg_toker
        self.max_len = max_len
        assert len(self.src) == len(self.trg)

    def __getitem__(self, idx):
        src_sent = self.src[idx]
        trg_sent = self.trg[idx]
        # src_dict = self.src_toker(src_sent, truncation=True, max_length=self.max_len, padding='max_length')
        src_ids = self.src_toker(src_sent, truncate=True).squeeze(dim=0)
        trg_dict = self.trg_toker(trg_sent, truncation=True, max_length=self.max_len, padding='max_length')
        # src_ids = src_dict['input_ids']
        # src_mask = src_dict['attention_mask']
        src_mask = [1]
        trg_ids = trg_dict['input_ids']
        trg_mask = trg_dict['attention_mask']
        token_type_ids = trg_dict['token_type_ids']
        
        # import pdb;pdb.set_trace()

        return src_ids, Tensor(src_mask), Tensor(trg_ids), Tensor(trg_mask), Tensor(token_type_ids), src_sent

        # return {
        #     'src_ids': src_ids,
        #     'src_mask': src_mask,
        #     'trg_ids': trg_ids,
        #     'trg_mask': trg_mask
        # }

    def __len__(self):
        return len(self.src)
    

class ImageTextDataset(Dataset):
    def __init__(self, names, json_file, image_dir, preprocess, toker=None, sentence_transformer=False, return_type='tuple',istrain=True,stage ="CMA"):
        self.names = np.load(names)
        self.data_dict = json.load(open(json_file, 'r', encoding='utf-8'))
        self.preprocess = preprocess
        self.toker = toker
        if self.toker ==  None:
            self.en_data_dict = json.load(open('dataset/ref_captions.json', 'r', encoding='utf-8'))
        
        self.return_type = return_type
        self.imgpath2imgid = {}
        self.istrain = istrain

        self.sentence_transformer = sentence_transformer
        self.stage = stage

        self.multi_sentence_per_video = True
        self.cut_off_points = []

        self.pairs = []
        not_found = []
        for name in self.names:
            if self.toker == None:
                if name not in self.en_data_dict:
                    continue
            
            if isinstance(name, np.bytes_):
                name = name.decode('utf-8')
            image_path = os.path.join(image_dir, name)

            if not os.path.isfile(image_path):
                not_found.append(name)
                continue
            self.imgpath2imgid[image_path] = name

            if self.toker == None:
                try:
                    for i, captions in enumerate(zip(self.data_dict[name], self.en_data_dict[name])):
                        caption = captions[0]
                        caption_en = captions[1]
                        self.pairs.append((image_path, caption_en, caption, i))
                except:
                    pass                    
            else:
                for i, caption in enumerate(self.data_dict[name]):
                    self.pairs.append((image_path, caption, i))

            self.cut_off_points.append(len(self.pairs))
        self.video_num = len(self.names)
        self.sentence_num = len(self.pairs)

        print(f'Total image-text pairs: {len(self.pairs)}')
        print(f'Not found image: {len(not_found)}')

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        if self.toker == None:
            image_path, caption_en, caption, i = self.pairs[idx]
        else:
            image_path, caption, i = self.pairs[idx]
        image_id = self.imgpath2imgid[image_path]
        if self.sentence_transformer:
            # img = Image.open(image_path)
            return image_path, image_path, caption
        if not self.istrain:
            img = self.preprocess(Image.open(image_path))
        elif self.stage == "CMA":
            img = self.preprocess(Image.open(image_path))
        # import pdb;pdb.set_trace()
        if self.toker:
            input_ids = self.toker(caption, truncate=True).squeeze(dim=0)
            if self.return_type == 'tuple':
                return img, input_ids, caption
            elif self.return_type == 'dict':
                return {
                    'img': img,
                    'image_id': image_id,
                    'input_ids': input_ids,
                    'caption': caption,
                    'caption_id': image_id+'#'+str(i)
                }
        else:
            if self.return_type == 'tuple': 
                if self.istrain and self.stage == "CLA":
                    return caption_en, caption
                else:
                    return img, caption_en, caption
            elif self.return_type == 'dict':
                return {
                    'img': img,
                    'image_id': image_id,
                    'caption': caption,
                    'caption_en': caption_en,
                    'caption_id': image_id+'#'+str(i)
                }
            
            
def create_tokenizer():
    model_name_or_path = './pretrained_model/bert-base-multilingual-cased'
    do_lower_case = True
    cache_dir = 'data/cache_dir'
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir)
    return tokenizer
caption_tokenizer = create_tokenizer()



class MultilingualImageTextDataset(Dataset):
    def __init__(self, names, json_files, langs, image_dir, preprocess, top=-1, train_en=False, stage='CLA', en2zh_path=''):
        self.names = np.load(names)
        self.data_dict = {}
        self.langs = langs
        self.stage = stage
        self.en2zh_path = en2zh_path
        self.trg_langs = [lang for lang in langs if lang != 'en']
        if train_en:
            self.trg_langs = self.trg_langs + ['en']
        print(f"Target languages: {self.trg_langs}")
        for lang, json_file in zip(langs, json_files):
            self.data_dict[lang] = json.load(open(json_file, 'r', encoding='utf-8'))
        if len(en2zh_path) != 0:
            self.en2zh = json.load(open(en2zh_path, 'r', encoding='utf-8'))
        not_found = set()
        self.id2path = {}
        #TODO please use your path
        all_image_names = set(json.load(open('dataset/ConceptualCaption/new_existing_ids.json')))
        for name in self.names:
            img_path = os.path.join(image_dir, name)
            # if not os.path.isfile(img_path):
            #     not_found.add(name)
            if name not in all_image_names:
                not_found.add(name)
            else:
                self.id2path[name] = img_path
        del all_image_names
        print(f'Images not found: {len(not_found)}')
        self.names = [name for name in self.names if name not in not_found]
        if top > 0:
            self.names = self.names[:top]
        self.preprocess = preprocess
        self.image_num = len(self.names)
        print(f'Total images: {self.image_num}')
        
    def __len__(self):
        return self.image_num
    
    def __getitem__(self, idx):
        img_id = self.names[idx]
        # lang = random.choice(self.langs)
        lang = random.choice(self.trg_langs)
        caption_trg = self.data_dict[lang][img_id][0]
        
        caption_en = self.data_dict['en'][img_id][0]
        image_path = self.id2path[img_id]
        if len(self.en2zh_path) != 0:
            caption_en2zh = self.en2zh[img_id][0]

        # if self.stage != 'CLA' and self.stage != 'fusion':
        # image_path = self.id2path[img_id]
        # img = self.preprocess(Image.open(image_path))

        if self.stage != 'CLA':
            image_path = self.id2path[img_id]
            
            img = self.preprocess(Image.open(image_path))

            return img, caption_en, caption_trg
        else:
            return caption_en, caption_trg ,caption_en2zh
       



def get_mit_loader(args, names, json_files, langs, image_dir, preprocess, is_train, top=-1, train_en=False, stage='CLA', en2zh_path=''):
    dataset = MultilingualImageTextDataset(names, json_files, langs, image_dir, preprocess, top=top, train_en=train_en,stage=stage,en2zh_path=en2zh_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.n_workers,
        drop_last=is_train
    )
    return dataset, dataloader

def get_it_loader(args, names, json_file, image_dir, preprocess, toker, is_train=True, st=False, return_type='tuple'):
    dataset = ImageTextDataset(
        names, json_file, image_dir, preprocess, toker, sentence_transformer=st, return_type=return_type, istrain=is_train
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.n_workers,
        drop_last=is_train
    )
    return dataset, dataloader
#new
def clean_str_cased(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()
def tokenize_caption(tokenizer, raw_caption, cap_id, type='EN'):

    if(type == 'EN'):
        word_list = clean_str_cased(raw_caption)
        txt_caption = " ".join(word_list)
        # Remove whitespace at beginning and end of the sentence.
        txt_caption = txt_caption.strip()
        # Add period at the end of the sentence if not already there.
        try:
            if txt_caption[-1] not in [".", "?", "!"]:
                txt_caption += "."
        except:
            print(cap_id)
        txt_caption = txt_caption.capitalize()

        #ids = tokenizer.encode(txt_caption, add_special_tokens=True,truncation=True,max_length=70)
        ids = tokenizer.encode(txt_caption, add_special_tokens=True)

    else:
        #ids = tokenizer.encode(raw_caption, add_special_tokens=True,truncation=True,max_length=70)
        ids = tokenizer.encode(raw_caption, add_special_tokens=True)

    return ids

class TxtDataSet4DualEncoding_translation2019():
    """
    Load captions
    """

    def __init__(self, args, cap_file, cap_file_zh, cap_file_trans2zh, cap_file_trans2en, lang_type):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split('\t', 1)

                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.length = len(self.cap_ids)
        print("paralel len", self.length)
        print(len(self.captions))
        # BERT
        self.tokenizer = caption_tokenizer# create_tokenizer()

        ##ZH
        self.captions_zh = {}
        self.cap_ids_zh = []
        with open(cap_file_zh, 'r') as zh_reader:
            for line in zh_reader.readlines():
                cap_id, caption = line.strip().split('\t', 1)

                self.captions_zh[cap_id] = caption
                self.cap_ids_zh.append(cap_id)
        
        # print(self.captions_zh)
        # sdasd
        # trans
        self.captions_trans2zh = {}
        self.cap_ids_trans2zh = []
        with open(cap_file_trans2zh, 'r') as trans_reader:
            for line in trans_reader.readlines():
                cap_id, caption = line.strip().split('\t', 1)
                self.captions_trans2zh[cap_id] = caption
                self.cap_ids_trans2zh.append(cap_id)

        self.captions_trans2en = {}
        self.cap_ids_trans2en = []
        with open(cap_file_trans2en, 'r') as trans_reader:
            for line in trans_reader.readlines():
                cap_id, caption = line.strip().split('\t', 1)
               
                self.captions_trans2en[cap_id] = caption
                self.cap_ids_trans2en.append(cap_id)

        self.type = lang_type

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        str_ls = cap_id.split('#')

        if self.type is None:
            tmp = '#zh#'
        elif 'zh2en' in self.type:
            tmp = '#zh2en#'
        elif 'en2zh' in self.type:
            tmp = '#en2zh#'
        else:
            tmp = '#zh#'

        cap_id_zh = str_ls[0] + "#zh#" + str_ls[2]
        cap_id_trans2zh = str_ls[0] + "#en2zh#" + str_ls[2]
        cap_id_trans2en = str_ls[0] + "#zh2en#" + str_ls[2]
        # BERT
        caption = self.captions[cap_id]
        bert_ids = tokenize_caption(self.tokenizer, caption, cap_id)
        bert_tensor = torch.Tensor(bert_ids)
        L = [x for x in range(1, len(bert_ids) - 1)]
        random = int(len(bert_ids) * 0.15)
        matrix = np.random.choice(L, random, replace=False)
        bert_ids_mask = copy.deepcopy(bert_ids)
        for j in range(len(matrix)):
            bert_ids_mask[matrix[j]] = 103
        bert_ids_mask = torch.Tensor(bert_ids_mask)
        # zh 
        caption_zh = self.captions_zh[cap_id_zh]
        bert_ids_zh = tokenize_caption(self.tokenizer, caption_zh, cap_id_zh, type='ZH')
        bert_tensor_zh = torch.Tensor(bert_ids_zh)
        L = [x for x in range(1, len(bert_ids_zh) - 1)]
        random = int(len(bert_ids_zh) * 0.15)
        matrix = np.random.choice(L, random, replace=False)
        bert_ids_zh_mask = copy.deepcopy(bert_ids_zh)
        for j in range(len(matrix)):
            bert_ids_zh_mask[matrix[j]] = 103
        bert_ids_zh_mask = torch.Tensor(bert_ids_zh_mask)

        # trans
        caption_trans2zh = self.captions_trans2zh[cap_id_trans2zh]
        bert_ids_trans2zh = tokenize_caption(self.tokenizer, caption_trans2zh, cap_id_trans2zh, type='ZH')
        bert_tensor_trans2zh = torch.Tensor(bert_ids_trans2zh)

        caption_trans2en = self.captions_trans2en[cap_id_trans2en]
        bert_ids_trans2en = tokenize_caption(self.tokenizer, caption_trans2en, cap_id_trans2en)
        bert_tensor_trans2en = torch.Tensor(bert_ids_trans2en)
        # return bert_tensor, bert_tensor_zh, bert_tensor_trans2zh, bert_tensor_trans2en, bert_ids_mask, bert_ids_zh_mask ,index, cap_id
        return caption ,caption_zh,caption_trans2en,caption_trans2zh
    def __len__(self):
        return self.length
class TxtDataSet4DualEncoding_translation2019_2():
    """
    Load captions
    """

    def __init__(self, args, cap_file, cap_file_zh,  lang_type):
        # Captions
        self.captions = {}
        self.cap_ids = []
        cap_id=0
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                caption = line.strip()
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                cap_id=cap_id+1
 
        self.length = len(self.cap_ids)
        print("paralel len", self.length)
        print(len(self.captions))
        # BERT
        self.tokenizer = caption_tokenizer# create_tokenizer()

        ##ZH
        self.captions_zh = {}
        self.cap_ids_zh = []
        cap_id=0
        with open(cap_file_zh, 'r') as zh_reader:
            for line in zh_reader.readlines():
                caption = line.strip()

                self.captions_zh[cap_id] = caption
                self.cap_ids_zh.append(cap_id)
                cap_id=cap_id+1
        
        # print(self.captions_zh)
        # sdasd
        # trans
        self.captions_trans2zh = {}
        self.cap_ids_trans2zh = []

        self.captions_trans2en = {}
        self.cap_ids_trans2en = []

        self.type = lang_type

    def __getitem__(self, index):

        # BERT
        caption = self.captions[index]
        # zh 
        caption_zh = self.captions_zh[index]

        return caption ,caption_zh
    def __len__(self):
        return self.length
    
class TxtDataSet4DualEncoding_translate():
    """
    Load captions
    """
    def __init__(self, args, cap_file, cap_file_zh, filter_trans_zh, lang_type):
        # Captions
        self.captions = {}
        self.cap_ids = []
        cap_id=0
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                caption = line.strip()
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                cap_id=cap_id+1
 
        self.length = len(self.cap_ids)
        print("paralel len", self.length)
        print(len(self.captions))
        # BERT
        self.tokenizer = caption_tokenizer# create_tokenizer()

        ##ZH
        self.captions_zh = {}
        self.cap_ids_zh = []
        cap_id=0
        with open(cap_file_zh, 'r') as zh_reader:
            for line in zh_reader.readlines():
                caption = line.strip()

                self.captions_zh[cap_id] = caption
                self.cap_ids_zh.append(cap_id)
                cap_id=cap_id+1

        ##filter_trans
        self.filter_trans_zh = {}
        self.filter_ids_zh = []
        cap_id=0
        with open(filter_trans_zh, 'r') as zh_filter_reader:
            for line in zh_filter_reader.readlines():
                caption = line.strip()

                self.filter_trans_zh[cap_id] = caption
                self.filter_ids_zh.append(cap_id)
                cap_id=cap_id+1

        
        # print(self.captions_zh)
        # sdasd
        # trans
        self.captions_trans2zh = {}
        self.cap_ids_trans2zh = []

        self.captions_trans2en = {}
        self.cap_ids_trans2en = []

        self.type = lang_type

    def __getitem__(self, index):

        # BERT
        caption = self.captions[index]
        # zh 
        caption_zh = self.captions_zh[index]
        #filter
        filter_trans_zh = self.filter_trans_zh[index]

        return caption ,caption_zh ,filter_trans_zh
    def __len__(self):
        return self.length

class TxtDataSet4DualEncoding_300k_translate():
    """
    Load captions
    """
    def __init__(self, args, cap_file, lang_type):
        # Captions
        self.captions = {}
        self.cap_ids = []
        cap_id=0
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                caption = line.strip()
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                cap_id=cap_id+1
 
        self.length = len(self.cap_ids)
        print("paralel len", self.length)
        print(len(self.captions))
        # BERT
        self.tokenizer = caption_tokenizer# create_tokenizer()

        
        # print(self.captions_zh)
        # sdasd
        # trans
        self.captions_trans2zh = {}
        self.cap_ids_trans2zh = []

        self.captions_trans2en = {}
        self.cap_ids_trans2en = []

        self.type = lang_type

    def __getitem__(self, index):

        # BERT
        caption = self.captions[index]

        return caption 
    def __len__(self):
        return self.length


def collate_parallel_text(data, args):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    # captions, cap_bows, idxs, cap_ids = zip(*data)
        bert_cap, bert_cap_zh, bert_cap_trans2zh, bert_cap_trans2en, bert_cap_mask, bert_cap_zh_mask,idxs, cap_ids = zip(*data)

    # BERT
    if bert_cap[0] is not None:
        lengths = [len(cap) for cap in bert_cap]
        bert_target = torch.zeros(len(bert_cap), max(lengths)).long()
        words_mask = torch.zeros(len(bert_cap), max(lengths))
        for i, cap in enumerate(bert_cap):
            end = lengths[i]
            bert_target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        bert_target = None
        words_mask = None
        lengths = None

    lengths = torch.IntTensor(lengths)
    # BERT
    text_data = (bert_target, lengths, words_mask)

    if bert_cap_mask[0] is not None:
        lengths = [len(cap) for cap in bert_cap_mask]
        bert_target_mask = torch.zeros(len(bert_cap_mask), max(lengths)).long()
        for i, cap in enumerate(bert_cap_mask):
            end = lengths[i]
            bert_target_mask[i, :end] = cap[:end]

    # BERT
    text_data_mask = (bert_target_mask, lengths, words_mask)

    #zh
    if bert_cap_zh[0] is not None:
        lengths_zh = [len(cap) for cap in bert_cap_zh]
        bert_target_zh = torch.zeros(len(bert_cap_zh), max(lengths_zh)).long()
        words_mask_zh = torch.zeros(len(bert_cap_zh), max(lengths_zh))
        for i, cap in enumerate(bert_cap_zh):
            end = lengths_zh[i]
            bert_target_zh[i, :end] = cap[:end]
            words_mask_zh[i, :end] = 1.0
    else:
        bert_target_zh = None
        words_mask_zh = None
        lengths_zh = None

    lengths_zh = torch.IntTensor(lengths_zh)
    # BERT
    text_data_zh = (bert_target_zh, lengths_zh, words_mask_zh)

    if bert_cap_zh_mask[0] is not None:
        lengths = [len(cap) for cap in bert_cap_zh_mask]
        bert_target_zh_mask = torch.zeros(len(bert_cap_zh_mask), max(lengths)).long()
        for i, cap in enumerate(bert_cap_zh_mask):
            end = lengths_zh[i]
            bert_target_zh_mask[i, :end] = cap[:end]


        # BERT
    text_data_zh_mask = (bert_target_zh_mask, lengths_zh, words_mask_zh)

    # trans
    if bert_cap_trans2zh[0] is not None:
        lengths_trans2zh = [len(cap) for cap in bert_cap_trans2zh]
        bert_target_trans2zh = torch.zeros(len(bert_cap_trans2zh), max(lengths_trans2zh)).long()
        words_mask_trans2zh = torch.zeros(len(bert_cap_trans2zh), max(lengths_trans2zh))
        for i, cap in enumerate(bert_cap_trans2zh):
            end = lengths_trans2zh[i]
            bert_target_trans2zh[i, :end] = cap[:end]
            words_mask_trans2zh[i, :end] = 1.0
    else:
        bert_target_trans2zh = None
        words_mask_trans2zh = None
        lengths_trans2zh = None
    lengths_trans2zh = torch.IntTensor(lengths_trans2zh)
    text_data_trans2zh = (bert_target_trans2zh, lengths_trans2zh, words_mask_trans2zh)




    if bert_cap_trans2en[0] is not None:
        lengths_trans2en = [len(cap) for cap in bert_cap_trans2en]
        bert_target_trans2en = torch.zeros(len(bert_cap_trans2en), max(lengths_trans2en)).long()
        words_mask_trans2en = torch.zeros(len(bert_cap_trans2en), max(lengths_trans2en))
        for i, cap in enumerate(bert_cap_trans2en):
            end = lengths_trans2en[i]
            bert_target_trans2en[i, :end] = cap[:end]
            words_mask_trans2en[i, :end] = 1.0
    else:
        bert_target_trans2en = None
        words_mask_trans2en = None
        lengths_trans2en = None
    lengths_trans2en = torch.IntTensor(lengths_trans2en)
    text_data_trans2en = (bert_target_trans2en, lengths_trans2en, words_mask_trans2en)




    text_data_all = text_data, text_data_zh, text_data_trans2zh, text_data_trans2en, text_data_mask, text_data_zh_mask
    return text_data_all, idxs, cap_ids



def get_textloader(args, paths, toker, max_len):
    datasets = []
    for path in paths:
        reader = TxtReader(path)
        datasets.append(TextDataset(reader, toker, max_len))
    dataset = ConcatDataset(datasets)
