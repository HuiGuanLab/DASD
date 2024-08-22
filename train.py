import os
import sys
from typing_extensions import OrderedDict

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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import normalize as norm
from torch.nn.utils import clip_grad_norm_
from dataloader import get_it_loader
from data.dataset_hub import get_cc, get_cc_iter, get_m30k, get_m30k_iter, get_coco_iter, get_coco, get_general_eval
from torch.nn import MSELoss
from optimization import BertAdam
import torch.nn.functional as F
from torch.optim import Adam
from evaluate import eval_epoch
from collections import OrderedDict

from torch.cuda.amp import GradScaler, autocast
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def convert_weights(model):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = MSELoss()
    
    def forward(self, src_feats, trg_feats, layers=False):
        loss = 0.
        if layers:
            for src_feat, trg_feat in zip(src_feats, trg_feats):
                loss += self.mse_loss(src_feat, trg_feat)
        else:
            loss = self.mse_loss(src_feats, trg_feats)
            # print(src_feats)
            # print(trg_feats)


        return loss


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

def finetune_clip(args, batch, clip_model, loss_nce, loss_mse, epoch, lang=None):
    loss = 0.
    batch = [t.cuda() if isinstance(t, torch.Tensor) else t for t in batch]
    loss_kd = 0.
    loss_lang_nce = 0.
    loss_itm_nce = 0.
    img, src_sents , trg_sents = batch
    img_feats = clip_model.encode_image(img)

    if args.finetune_all and lang != 'en':
        if args.new_embed:
            if args.m_acquirer:
                txt_feats = clip_model.encode_text(src_sents, acquirer=True, tokenize=True, layers=args.kd_layers, lang=lang)
            else:
                txt_feats = clip_model.encode_text(src_sents, acquirer=True, tokenize=True, layers=args.kd_layers)
        else:
            txt_feats = clip_model.encode_text(clip.tokenize(list(src_sents), truncate=True).cuda(), acquirer=True, layers=args.kd_layers)
    else:
        txt_feats = clip_model.encode_text(src_sents, tokenize=True)

    sim_matrix = 100 * torch.matmul(norm(img_feats, dim=-1, eps=1e-6), norm(txt_feats, dim=-1, eps=1e-6).t())
    loss1 = loss_nce(sim_matrix)
    loss2 = loss_nce(sim_matrix.T)
    loss_itm_nce = (loss1 + loss2) / 2
    loss += loss_itm_nce    

    return {
        'loss': loss,
        'loss_kd': loss_kd,
        'loss_lang_nce': loss_lang_nce,
        'loss_itm_nce': loss_itm_nce
    }

def train_video(args, batch, model, loss_nce, loss_mse, epoch, lang=None):
    loss = 0
    batch = [t.cuda() if isinstance(t, torch.Tensor) else t for t in batch]
    sents, video, video_mask = batch
    vid_feats = model.get_visual_output(video, video_mask)
    if lang != 'en':
        txt_feats = model.get_sequence_output(sents, acquirer=True, lang=lang)
    else:
        txt_feats = model.get_sequence_output(sents)
    
    sim_matrix = 100 * torch.matmul(vid_feats, txt_feats.t())
    loss1 = loss_nce(sim_matrix)
    loss2 = loss_nce(sim_matrix.T)
    loss = (loss1 + loss2) / 2
    return {
        'loss': loss,
    }


def train_acquirer(args, batch, clip_model, loss_nce, loss_mse, epoch, lang=None):
    loss = 0.
    loss_kd = 0.
    loss_lang_nce = 0.
    loss_itm_nce = 0.
    loss_itm_mse = 0.
    loss_distill=0.
    density_loss=0.
    loss_mu= 0.
    adv_loss = 0.
    loss_teacher = 0.
    batch = [t.cuda() if isinstance(t, torch.Tensor) else t for t in batch]
    if args.stage == 'CLA':

        if len(batch) == 2:
            src_sents, trg_sents = batch
            with torch.no_grad():
                src_feats = clip_model.encode_text(clip.tokenize(list(src_sents), truncate=True).cuda(), acquirer=False, layers=args.kd_layers)
        elif len(batch) == 3:
            src_sents, trg_sents, trans_sents = batch
            with torch.no_grad():
                src_feats = clip_model.encode_text(clip.tokenize(list(src_sents), truncate=True).cuda(), acquirer=False, layers=args.kd_layers, lang='en',istrain=True)
    else:
        if len(batch) == 2:
            img, trg_sents = batch
            
            src_sents = None
        elif len(batch) == 3:
            img, src_sents, trg_sents = batch
            with torch.no_grad():
                src_feats = clip_model.encode_text(clip.tokenize(list(src_sents), truncate=True).cuda(), acquirer=False, layers=args.kd_layers)
        
    if args.new_embed:
        if args.m_acquirer:
            if args.stage == 'CLA':
                gold_feats = src_feats.detach()
            else:
                with torch.no_grad():
                    img_feats = clip_model.encode_image(img).cuda().float().detach()
                gold_feats = img_feats
            sr, sa, adv_loss,src_feats_distill = clip_model.encode_text(trg_sents, acquirer=True, tokenize=True, layers=args.kd_layers, lang=lang, zi_bool=True,src_feats=gold_feats,istrain=True)
            trg_feats = clip_model.encode_text(trg_sents, acquirer=True, tokenize=True, layers=args.kd_layers, lang=lang, sr=sr, sa=sa)

        else:
            trg_feats = clip_model.encode_text(trg_sents, acquirer=True, tokenize=True, layers=args.kd_layers)
    else:
        trg_feats = clip_model.encode_text(clip.tokenize(list(trg_sents), truncate=True).cuda(), acquirer=True, layers=args.kd_layers, lang=lang)

    if args.mse:
        loss_kd = loss_mse(src_feats, trg_feats, layers=(args.kd_layers is not None and epoch < args.kd_layer_ep))

        loss_distill = nn.SmoothL1Loss()
        loss_mu = loss_distill(sr, src_feats) * 0.1

        loss += loss_kd + adv_loss + loss_mu
    if args.nce:
        sim_matrix = torch.matmul(norm(src_feats, dim=-1), norm(trg_feats, dim=-1).t())

        sim_matrix = 100 * sim_matrix
        loss1 = loss_nce(sim_matrix)
        loss2 = loss_nce(sim_matrix.T)
        loss_lang_nce = (loss1 + loss2) / 2
        loss += loss_lang_nce
        
    if args.itm_nce:
        with torch.no_grad():
            img_feats = clip_model.encode_image(img).cuda().float().detach()
        sim_matrix = 100 * torch.matmul(norm(img_feats, dim=-1, eps=1e-6), norm(trg_feats, dim=-1, eps=1e-6).t())
        loss1 = loss_nce(sim_matrix)
        loss2 = loss_nce(sim_matrix.T)
        loss_itm_nce = (loss1 + loss2) / 2
        
        loss_distill = nn.SmoothL1Loss()
        loss_mu = loss_distill(img_feats, sr)  * 0.1

        loss_kd = loss_mse(src_feats, trg_feats, layers=(args.kd_layers is not None and epoch < args.kd_layer_ep))
        loss += loss_itm_nce + loss_mu + adv_loss + loss_kd
    if args.itm_mse:
        with torch.no_grad():
            img_feats = clip_model.encode_image(img).cuda().float().detach()
        loss_itm_mse = loss_mse(trg_feats, img_feats)
        loss += loss_itm_mse

    return {
        'loss': loss,
        'loss_kd': loss_kd,
        'loss_lang_nce': loss_lang_nce,
        'loss_itm_nce': loss_itm_nce,
        'loss_itm_mse': loss_itm_mse,
        'density_loss':density_loss,
        'loss_mu':loss_mu,
        'adv_loss':adv_loss,
    }

def merge_dict(path_dict_1, path_dict_2):
    if isinstance(path_dict_1, list):
        assert isinstance(path_dict_2, list)
        return path_dict_1 + path_dict_2
    merged = OrderedDict()
    for key in path_dict_1.keys():
        merged[key] = merge_dict(path_dict_1[key], path_dict_2[key])
    return merged


def main(args):
    set_seed(args)

    writer = SummaryWriter(log_dir=args.output_dir)
    logger = get_logger(filename=args.output_dir+'log.txt')
    logger.info('Config:')
    logger.info(json.dumps(args.__dict__, indent=1, ensure_ascii=False))

    clip_ckpt = args.clip_ckpt
    trn_langs = [l for l in args.langs.split(',') if l != 'en']
    clip_model, preprocess = clip.load(clip_ckpt, device='cuda', acquirer=(not args.finetune or args.finetune_all), jit=False, d_acquirer_hidden=args.acquirer_hidden, m_acquirer=args.m_acquirer, langs=(trn_langs if args.m_acquirer else None), skip=args.skip, init_mbert_embedding=args.init_mbert_embedding,stage=args.stage)
    
    
    # with open('parameters.txt', 'w') as file:
    #     file.write(clip_model.keys())
    # a
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
        
    clip_model = clip_model.cuda().float().train()
    # convert_weights(clip_model)
    # stat(clip_model, (3, 224, 224))
    total_params = sum(p.numel() for p in clip_model.parameters())

    print(f"Total number of parameters: {total_params}")
    src_toker = clip.tokenize
    trg_toker = transformers.AutoTokenizer.from_pretrained('./pretrained_model/bert-base-multilingual-cased')
    if args.img_type == 'mit_cc':
        if args.iter:
            train_dataloader = get_cc_iter(args, preprocess, is_train=True, langs=args.langs.split(','), top=args.top, train_en=False,stage=args.stage)
        else:
            train_dataset, train_dataloader = get_cc(args, preprocess, is_train=True, langs=args.langs.split(','), top=args.top)
    if args.img_type == 'm30k':
        if args.iter:
            train_dataloader = get_m30k_iter(args, preprocess, 'train', langs=args.langs.split(','))
        else:
            train_dataset, train_dataloader = get_m30k(args, preprocess, 'train', langs=['en'])
    if args.img_type == 'coco':
        if args.iter:
            train_dataloader = get_coco_iter(args, preprocess, 'train', langs=args.langs.split(','))
        else:
            train_dataset, train_dataloader = get_coco(args, preprocess, 'train', langs=['en'])    
    if args.img_type == 'mclip_cc':
        if args.iter:
            train_dataloader = get_mclip_cc_iter(args, preprocess, is_train=True, langs=args.langs.split(','))
        else:
            raise NotImplementedError

    logger.info('Using BertAdam optimizer')
    if args.iter:
        num_train_optimization_steps = args.max_step
        args.epochs = 1
    else:
        num_train_optimization_steps = int(len(train_dataloader)) * args.epochs

    param_optimizer = list(clip_model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.finetune or args.finetune_all or args.train_all:
        decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    else:
        if args.fix_embed:
            train_names = ['acquirer']
        elif args.fix_embed_tune_linear:
            train_names = ['acquirer', 'multilingual_embedding_linear']
        elif args.fix_linear:
            train_names = ['acquirer', 'multilingual_embedding.']
        else:
            train_names = ['acquirer', 'multilingual_embedding', 'multilingual_embedding_linear','ra','sa','zi','AdvAgent']
            # train_names = ['acquirer', 'multilingual_embedding', 'multilingual_embedding_linear']
            
        logger.info(f'Trainable parameter keys:{[n for n, p in param_optimizer if any(nd in n for nd in train_names)]}')
        decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any(nd in n for nd in train_names)]
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in train_names)]          

    weight_decay = 0.2
    if args.txt_lr:
        decay_vis_param_tp = [(n, p) for n, p in decay_param_tp if 'visual.' in n]
        no_decay_vis_param_tp = [(n, p) for n, p in no_decay_param_tp if 'visual.' in n]
        decay_no_vis_param_tp = [(n, p) for n, p in decay_param_tp if 'visual.' not in n]
        no_decay_no_vis_param_tp = [(n, p) for n, p in no_decay_param_tp if 'visual.' not in n]

        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_vis_param_tp], 'weight_decay': weight_decay, 'lr': args.lr},
            {'params': [p for n, p in no_decay_vis_param_tp], 'weight_decay': 0.0},
            {'params': [p for n, p in decay_no_vis_param_tp], 'weight_decay': weight_decay, 'lr': args.txt_lr},
            {'params': [p for n, p in no_decay_no_vis_param_tp], 'weight_decay': 0.0, 'lr': args.txt_lr},
        ]   
    elif args.acquirer_lr:
        ada_names = ['acquirer', 'multilingual_embedding', 'multilingual_embedding_linear']
        decay_ada_param_tp = [(n, p) for n, p in decay_param_tp if any(ada_k in n for ada_k in ada_names)]
        no_decay_ada_param_tp = [(n, p) for n, p in no_decay_param_tp if any(ada_k in n for ada_k in ada_names)]
        decay_no_ada_param_tp = [(n, p) for n, p in decay_param_tp if not any(ada_k in n for ada_k in ada_names)]
        no_decay_no_ada_param_tp = [(n, p) for n, p in no_decay_param_tp if not any(ada_k in n for ada_k in ada_names)]
        logger.info(f'Use different lr for acquirers: {args.acquirer_lr}')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_ada_param_tp], 'weight_decay': weight_decay, 'lr': args.acquirer_lr},
            {'params': [p for n, p in no_decay_ada_param_tp], 'weight_decay': 0.0, 'lr': args.acquirer_lr},
            {'params': [p for n, p in decay_no_ada_param_tp], 'weight_decay': weight_decay, 'lr': args.lr},
            {'params': [p for n, p in no_decay_no_ada_param_tp], 'weight_decay': 0.0, 'lr': args.lr},
        ]        
    elif args.embedding_lr:
        emb_names = ['multilingual_embedding', 'multilingual_embedding_linear']
        decay_ada_param_tp = [(n, p) for n, p in decay_param_tp if any(emb_k in n for emb_k in emb_names)]
        no_decay_ada_param_tp = [(n, p) for n, p in no_decay_param_tp if any(emb_k in n for emb_k in emb_names)]
        decay_no_ada_param_tp = [(n, p) for n, p in decay_param_tp if not any(emb_k in n for emb_k in emb_names)]
        no_decay_no_ada_param_tp = [(n, p) for n, p in no_decay_param_tp if not any(emb_k in n for emb_k in emb_names)]

        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_ada_param_tp], 'weight_decay': weight_decay, 'lr': args.embedding_lr},
            {'params': [p for n, p in no_decay_ada_param_tp], 'weight_decay': 0.0, 'lr': args.embedding_lr},
            {'params': [p for n, p in decay_no_ada_param_tp], 'weight_decay': weight_decay, 'lr': args.lr},
            {'params': [p for n, p in no_decay_no_ada_param_tp], 'weight_decay': 0.0, 'lr': args.lr},
        ]              
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_param_tp], 'weight_decay': weight_decay, 'lr': args.lr},
            {'params': [p for n, p in no_decay_param_tp], 'weight_decay': 0.0}
        ]   

    if args.optim == 'BertAdam':
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=0.1,
                            schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                            t_total=num_train_optimization_steps, weight_decay=weight_decay,
                            max_grad_norm=1.0)
    elif args.optim == 'Adam':
        optimizer = Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-6)

    global_step = 0
    
    loss_fct = MSE()
    loss_nce = CrossEn()

    val_loaders = []
    test_loaders = []
    if args.val_dataset == 'general':
        val_loaders, val_langs = get_general_eval(args, args.path_file, preprocess, src_toker, 'val')
        test_loaders, _ = get_general_eval(args, args.path_file, preprocess, src_toker, 'test')
    else:
        if args.val_dataset == 'm30k':
            path_dict = json.load(open(os.path.join(args.data_path,'Multi30k/multi30k_4lang_path.json')))
        elif args.val_dataset == 'coco':
            path_dict = json.load(open(os.path.join(args.data_path,'MSCOCO/COCO_en_split0_path.json')))
        elif args.val_dataset == 'm30k+coco':
            path_dict_m30k = json.load(open(os.path.join(args.data_path,'Multi30k/multi30k_4lang_path.json')))
            path_dict_coco = json.load(open(os.path.join(args.data_path,'MSCOCO/COCO_en_split0_path.json')))
            path_dict = merge_dict(path_dict_m30k, path_dict_coco)
        elif args.val_dataset == 'mclip_cc':
            path_dict = json.load(open(os.path.join(args.data_path,'./mclip_cc_6lang.json')))

        anno_files = path_dict['val']['anno_file']
        image_dirs = path_dict['val']['feature']
        name_files = path_dict['val']['name_file']
        langs = path_dict['val']['langs']
        config_val_langs = args.langs.split(',')
        if config_val_langs[0] == 'en' and not args.finetune_all:
            config_val_langs = config_val_langs[1:]
        val_langs = []
        if args.eval_epoch:
            for anno, image_dir, name_file, lang in zip(anno_files, image_dirs, name_files, langs):
                if lang not in config_val_langs:
                    continue
                _, loader = get_it_loader(args, name_file, anno, image_dir, preprocess, src_toker, is_train=False)
                val_loaders.append(loader)
                val_langs.append(lang)

            test_anno_files = path_dict['test']['anno_file']
            test_image_dirs = path_dict['test']['feature']
            test_name_files = path_dict['test']['name_file']
            test_langs = path_dict['test']['langs']

            for anno, image_dir, name_file, lang in zip(test_anno_files, test_image_dirs, test_name_files, test_langs):
                if lang not in config_val_langs:
                    continue
                _, loader = get_it_loader(args, name_file, anno, image_dir, preprocess, src_toker, is_train=False)
                test_loaders.append(loader)

    eval_epoch(clip_model, test_loaders, 'cuda', 1, val_langs, trg_toker, logger, is_clip=True, acquirer=True, new_embed=args.new_embed, extra_embed=args.extra_embed)
    best_score = 0.0
    best_epoch = 0
    best_ckpt_path = ''

    scaler = GradScaler()
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        clip_model.train()
        total_loss = 0
        epoch_step = args.max_step if args.iter else len(train_dataloader)
        # file = open('prefix_encoder.txt', 'w') 
        for step, batch in enumerate(tqdm(train_dataloader, total=epoch_step, ncols=50)):
            # with torch.autograd.set_detect_anomaly(True):
                if args.iter and step > args.max_step:
                    break
                lang = None
                if args.iter:
                    lang, batch = batch
                    
                if args.fp16:
                    with autocast():
                        if args.finetune:
                            loss_dict = finetune_clip(args, batch, clip_model, loss_nce, loss_fct, epoch, lang)
                        else:
                            loss_dict = train_acquirer(args, batch, clip_model, loss_nce, loss_fct, epoch, lang)
                        loss = loss_dict['loss']
                        
                        if torch.any(torch.isnan(loss)):

                            logger.info('Loss encounter NaN, break')
                            break
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.finetune:
                        loss_dict = finetune_clip(args, batch, clip_model, loss_nce, loss_fct, epoch, lang)
                    else:
                        loss_dict = train_acquirer(args, batch, clip_model, loss_nce, loss_fct, epoch, lang)
                    loss = loss_dict['loss']
                    optimizer.zero_grad()
                    loss_dict['loss'].backward()
                    clip_grad_norm_(clip_model.parameters(), 1.0)
                    optimizer.step()
                
                total_loss += float(loss_dict['loss'])
                if (step + 1) % 100 == 0:
                    if args.optim == 'BertAdam':
                        min_lr = float(min(list(set(optimizer.get_lr()))))
                    else:
                        min_lr = args.lr
                    logger.info(f'Epoch: {epoch+1}/{args.epochs}, Step: {step+1}/{epoch_step}, Running Loss: {loss_dict["loss"].float()}, lr: {min_lr}')
                    for loss_name, loss_tensor in loss_dict.items():
                        if loss_name == 'loss':
                            continue
                        logger.info(f'Loss {loss_name}: {float(loss_tensor)}')
                        writer.add_scalar(loss_name, float(loss_tensor), global_step=global_step)
                    writer.add_scalar('running loss', float(loss_dict['loss']), global_step=global_step)
                    writer.add_scalar('lr', float(min_lr), global_step=global_step)

                global_step += 1

                if args.eval_step > 0 and global_step % args.eval_step == 0:
                    mAR = eval_epoch(clip_model, val_loaders, 'cuda', 1, val_langs, trg_toker, logger, is_clip=True, acquirer=True, new_embed=args.new_embed, extra_embed=args.extra_embed)
                    torch.cuda.empty_cache()
                    clip_model.train()
                    writer.add_scalar('metrics/mAR_step', mAR, global_step=global_step)
                    if mAR > best_score:
                        best_score = mAR
                        best_epoch = global_step
                        best_ckpt_path = args.output_dir+'/models/'+'/pytorch_model.bin.{}'.format(best_epoch)
                        ###
                        if args.stage == 'CLA':
                            file_path = 'expr/vitb32/CMA/config.json' 
                            new_clip_ckpt_value = best_ckpt_path
                            with open(file_path, 'r') as file:
                                data = json.load(file)
                            data['clip_ckpt'] = new_clip_ckpt_value
                            with open(file_path, 'w') as file:
                                json.dump(data, file, indent=4)  
                        ###
                        os.makedirs(args.output_dir+'/models/', exist_ok=True)
                        torch.save(clip_model.state_dict(), args.output_dir+'/models/'+'/pytorch_model.bin.{}'.format(best_epoch))
                    logger.info(f'Best model at: {best_ckpt_path}, mAR: {best_score}')

        
        if not args.iter:
            total_loss = total_loss / len(train_dataloader)
            writer.add_scalar('total loss', total_loss, global_step=epoch+1)
            logger.info(f'Epoch: {epoch+1}/{args.epochs}, Total Loss: {total_loss}')
        os.makedirs(args.output_dir+'/models/', exist_ok=True)
        torch.save(clip_model.state_dict(), args.output_dir+'/models/'+'/pytorch_model.bin.{}'.format(epoch+1))
        
        if args.eval_epoch:
            mAR = eval_epoch(clip_model, test_loaders, 'cuda', 1, val_langs, trg_toker, logger, is_clip=True, 
            acquirer=True, new_embed=args.new_embed, extra_embed=args.extra_embed)
            writer.add_scalar('metrics/mAR', mAR, global_step=epoch+1)
            consecutive_not_improve = 0
            if mAR > best_score:
                consecutive_not_improve = 0
                best_score = mAR
                best_epoch = epoch+1
                os.makedirs(args.output_dir+'/models/', exist_ok=True)
                best_ckpt_path = args.output_dir+'/models/'+'/pytorch_model.bin.{}'.format(best_epoch)
            else:
                consecutive_not_improve += 1
            logger.info(f'Best model at {best_ckpt_path}, mAR: {best_score}')


    if args.eval_epoch:
        torch.cuda.empty_cache()
        checkpoint = torch.load(best_ckpt_path, map_location='cuda')
        logger.info(f'Resumed from {best_ckpt_path} for testing...')
        clip_model.load_state_dict(checkpoint, strict=False)
        mAR = eval_epoch(clip_model, test_loaders, 'cuda', 1, val_langs, trg_toker, logger, is_clip=True,
        acquirer=True, new_embed=args.new_embed, extra_embed=args.extra_embed)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    
    parser.add_argument('--finetune', action='store_true', help='finetune CLIP on english dataset')
    parser.add_argument('--finetune_all', action='store_true', help='finetune CLIP and acquirer on all language')
    parser.add_argument('--train_all', action='store_true', help='for train all experiments')

    parser.add_argument('--path_file', type=str, default=None, help='data path file')

    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--txt_lr', default=None, type=float)
    parser.add_argument('--acquirer_lr', default=None, type=float)
    parser.add_argument('--embedding_lr', default=None, type=float)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--early_stopping', default=-1, type=int, help='if `early_stopping` consecutive validations do not improve, the training stops')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--optim', default='Adam', choices=['BertAdam', 'Adam'])
    parser.add_argument('--mse', action='store_true')
    parser.add_argument('--nce', action='store_true')
    parser.add_argument('--itm_nce', action='store_true')
    parser.add_argument('--itm_mse', action='store_true')
    parser.add_argument('--img_type', choices=['cc300k_r', 'cc', 'cc300k', 'm30k', 'coco'], type=str, default='mit_cc')
    parser.add_argument('--val_dataset', choices=['m30k', 'coco', 'm30k+coco'], default='m30k+coco')
    parser.add_argument('--top', type=int, default=-1)
    parser.add_argument('--acquirer_hidden', type=int, default=256)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--clip_ckpt', type=str, default='pretrained_model/clip/ViT-B-32.pt')
    parser.add_argument('--acquirer_ckpt', type=str, default=None)
    parser.add_argument('--embedding_ckpt', type=str, default=None)

    parser.add_argument('--new_embed', action='store_true')
    parser.add_argument('--extra_embed', action='store_true')
    parser.add_argument('--init_mbert_embedding', action='store_false')
    parser.add_argument('--fix_embed', action='store_true')
    parser.add_argument('--fix_linear', action='store_true')
    parser.add_argument('--fix_embed_tune_linear', action='store_true')

    parser.add_argument('--m_acquirer', action='store_true')
    parser.add_argument('--skip', action='store_true')
    
    parser.add_argument('--iter', action='store_true')
    parser.add_argument('--max_step', type=int, default=-1)

    parser.add_argument('--kd_layers', default=None, type=list)
    parser.add_argument('--kd_layer_ep', default=50, type=int)


    parser.add_argument('--langs', type=str, default='en,zh')
    parser.add_argument('--eval_langs', type=str, default=None)

    parser.add_argument('--model', type=str, default='clip')

    parser.add_argument('--video', action='store_true')
    parser.add_argument('--ratio', default=None)
    parser.add_argument('--data_path', type=str,default=None)
    parser.add_argument('--stage', type=str,default=None)

    args = parser.parse_args()
    
    if args.eval_langs is None:
        args.eval_langs = args.langs

    if args.config is not None:
        args_dict = json.load(open(args.config, 'r', encoding='utf-8'))
        for key, value in args_dict.items():
            setattr(args, key, value)

    os.makedirs(args.output_dir+'/opt/', exist_ok=True)
    with open(f'{args.output_dir}/opt/config.json', 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, indent=1, ensure_ascii=False)
        print(args.__dict__)


    main(args)
