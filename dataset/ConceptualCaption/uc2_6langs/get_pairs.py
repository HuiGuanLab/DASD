import json
import numpy as np
import os

langs = ['en', 'de', 'fr', 'cs', 'zh', 'ja']
ref_caption_dict = {}

names_set = set()
for lang in langs:
    with open(f'jsons/ref_captions_{lang}.json', 'r', encoding='utf-8') as f:
        ref_captions = json.load(f)
    if len(names_set) == 0:
        names_set = set(ref_captions.keys())
    else:
        names_set = names_set & set(ref_captions.keys())

names = np.array(sorted(list(names_set)))

np.save('names_aligned.npy', names)
    

'''
for trg_lang in trg_langs:
    en_captions = []
    trg_captions = []
    with open('jsons/ref_captions_{}.json'.format(trg_lang), 'r', encoding='utf-8') as f:
        ref_trg = json.load(f)
    for img_id, captions in ref_en.items():
        if img_id not in ref_trg:
            continue
        en_caption = captions[0]
        trg_caption = ref_trg[img_id][0]
        en_captions.append(en_caption+'\n')
        trg_captions.append(trg_captions+'\n')
    with open('')
'''
