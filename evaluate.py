import os
import clip
import torch
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import json
from PIL import Image

from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

def obj_to_device(obj, device='cuda'):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [obj_to_device(t, device) for t in obj]
    elif isinstance(obj, dict):
        return {key: obj_to_device(t, device) for key, t in obj.items()}
    else:
        return obj


def get_similarity_logits(sequence_output, visual_output):
    sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
    return 100 * torch.matmul(sequence_output, visual_output.t())

def _run_on_single_gpu(batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in tqdm(enumerate(batch_list_t), total=len(batch_list_t), ncols=50):
        # input_mask, segment_ids = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, visual_output in enumerate(batch_visual_output_list):
            # video_mask, *_tmp = b2
            # visual_output = [idx2]
            # b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
            #                                                          loose_type=model.loose_type)
            b1b2_logits = get_similarity_logits(sequence_output, visual_output)             
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def average_dict(metric_dict_list):
    n = len(metric_dict_list)
    average_metrics = OrderedDict()
    for key in metric_dict_list[0].keys():
        if isinstance(metric_dict_list[0][key], list):
            continue
        average_metrics[key] = sum([metrics[key] for metrics in metric_dict_list]) * 1.0 / n
    return average_metrics


def eval_epoch(clip_model, test_dataloader, device, n_gpu, langs=['en'], 
            toker=None, logger=None, is_clip=False, return_all_metrics=False, 
            sentence_transformer=False, acquirer=False, new_embed=False, 
            extra_embed=False, save_matrix=False, output_root=None):
    if isinstance(test_dataloader, list):
        assert len(test_dataloader) == len(langs)
        metric = []
        for dataloader, lang in zip(test_dataloader, langs):
            lang_ar = eval_epoch(clip_model, dataloader, device, n_gpu, [lang], 
                            toker, logger, is_clip, return_all_metrics, sentence_transformer, 
                            acquirer=acquirer, new_embed=new_embed, 
                            extra_embed=extra_embed, save_matrix=save_matrix, 
                            output_root=(None if output_root is None else os.path.join(output_root, lang)))
            metric.append(lang_ar)
        logger.info("Text-to-Video:")
        if not return_all_metrics:
            for lang, ar in zip(langs, metric):
                logger.info('{} AR: {:.1f}'.format(lang, ar))
            mAR = sum(metric) / len(metric)
            logger.info('AR: {:.1f}'.format(mAR))
            return mAR
        else:
            for lang, me in zip(langs, metric):
                logger.info('######### Lang {} #########'.format(lang))
                for name, value in me.items():
                    if isinstance(value, list):
                        continue
                    logger.info('{}: {:.1f}'.format(name, value))

            meta_average_metrics = average_dict(metric)
            logger.info('######### Average #########'.format(lang))
            for name, value in meta_average_metrics.items():
                logger.info('{}: {:.1f}'.format(name, value))
            
            return meta_average_metrics

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    clip_model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        if save_matrix:
            all_caption_ids = []
            all_image_ids = []

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        visual_json = {}
        for bid, batch in enumerate(test_dataloader):
            # batch = tuple(t.to(device) for t in batch if isinstance(t, torch.Tensor))
            if isinstance(batch, tuple) or isinstance(batch, list):
                batch = tuple((t.to(device) if isinstance(t, torch.Tensor) else t) for t in batch)
                # input_ids, input_mask, segment_ids, video, video_mask = batch
                images, input_ids, captions = batch
            elif isinstance(batch, dict):
                batch = obj_to_device(batch, 'cuda')
                images = batch['img']
                input_ids = batch['input_ids']
                captions = batch['caption']
                caption_id = batch['caption_id']
            
            if save_matrix:
                all_caption_ids.extend(list(caption_id))
                all_image_ids.extend(list(batch['image_id']))


            if not sentence_transformer:
                input_dict = toker(list(captions), padding=True, return_tensors='pt', truncation=True, max_length=512)
                input_ids = input_dict['input_ids'].cuda()
                attention_mask = input_dict['attention_mask'].cuda()
                token_type_ids = input_dict['token_type_ids'].cuda()

            # multi-sentences retrieval means: one clip has two or more descriptions.
            b = len(images)
            if new_embed:
                test_lang = None if langs[0].split('_')[0]=='en' else langs[0].split('_')[0]
                sr,sa = clip_model.encode_text(list(captions), acquirer=True, tokenize=True, lang=test_lang,zi_bool=True)
                sequence_output = clip_model.encode_text(list(captions), acquirer=(acquirer and langs[0].split('_')[0]!='en'), tokenize=True, lang=test_lang,sr=sr,sa=sa).float()
            elif extra_embed:
                sequence_output = clip_model.encode_text(list(captions), acquirer=(acquirer and langs[0].split('_')[0]!='en'), tokenize=True, m_toker=toker).float()
            else:
                test_lang = None if langs[0].split('_')[0]=='en' else langs[0].split('_')[0]
                text = clip.tokenize(list(captions), truncate=True).to('cuda')
                sequence_output = clip_model.encode_text(text, acquirer=(acquirer and langs[0].split('_')[0]!='en'), lang=test_lang).float()

            batch_sequence_output_list.append(sequence_output)
            # batch_list_t.append((attention_mask, token_type_ids,))
            batch_list_t = batch_sequence_output_list
            # batch_list_t = None

            s_, e_ = total_video_num, total_video_num + b
            filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

            if len(filter_inds) > 0:
                if sentence_transformer:
                    images = [images[i] for i in filter_inds]
                else:
                    images = images[filter_inds, ...]
                if sentence_transformer:
                    visual_output = torch.from_numpy(clip_model.encode(images)).cuda()
                else:    
                    # import pdb;pdb.set_trace()
                    visual_output = clip_model.encode_image(images).float()
                batch_visual_output_list.append(visual_output)
            total_video_num += b

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        sim_matrix = _run_on_single_gpu(batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = np.concatenate(sim_matrix, axis=0)
        with open("feature.json","a",encoding="utf-8") as f:
            json.dump(visual_json,f,ensure_ascii=False)
    if save_matrix:
        all_image_ids = [all_image_ids[i] for i in cut_off_points_]
        os.makedirs(output_root, exist_ok=True)
        with open(os.path.join(output_root, '0_caption_ids.json'), 'w', encoding='utf-8') as f:
            json.dump(all_caption_ids, f, indent=1, ensure_ascii=False)
        with open(os.path.join(output_root, '1_image_ids.json'), 'w', encoding='utf-8') as f:
            json.dump(all_image_ids, f, indent=1, ensure_ascii=False)
        np.save(os.path.join(output_root, 'sim_matrix.npy'), sim_matrix)

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
    average_recall_tv = (tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10']) / 3
    average_recall_vt = (vt_metrics['R1'] + vt_metrics['R5'] + vt_metrics['R10']) / 3
    average_recall = (average_recall_tv + average_recall_vt) / 2
    logger.info("Average Recall:")
    logger.info('\t>>>  T2V$AR: {:.1f} - V2T$AR: {:.1f} - AR: {:.1f}'.format(average_recall_tv, average_recall_vt, average_recall))    
    R1 = tv_metrics['R1']
    if not return_all_metrics:
        return average_recall
    else:
        all_metrics = OrderedDict()
        for name, value in tv_metrics.items():
            all_metrics['T2V_'+name] = value
        for name, value in vt_metrics.items():
            all_metrics['V2T_'+name] = value
        all_metrics['T2V_AR'] = average_recall_tv
        all_metrics['V2T_AR'] = average_recall_vt
        all_metrics['AR'] = average_recall
    
        return all_metrics
        