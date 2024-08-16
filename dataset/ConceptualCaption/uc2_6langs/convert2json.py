import json
from collections import OrderedDict
from tqdm import tqdm

langs = ['en', 'de', 'fr', 'cs', 'zh', 'ja', 'ko']




for lang in langs:
    with open(f'raw/train_imageId2Ann_{lang}.tsv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ref_captions = OrderedDict()
    for line in tqdm(lines, total=len(lines)):
        line = line.strip()
        _id, img_url, caption, state = line.split('\t')
        if state not in ['success', 'fail']:
            import pdb;pdb.set_trace()
        if state == 'fail':
            continue
        if caption == '' and state == 'success':
            import pdb;pdb.set_trace()
        if caption != '' and state == 'fail':
            import pdb;pdb.set_trace()

        if caption == '':
            continue
        img_id = '%08d.jpg' % (int(_id))
        if img_id not in ref_captions:
            ref_captions[img_id] = []

        ref_captions[img_id].append(
            caption
        )
    print(f'Lang {lang} has {len(ref_captions)} images')
    with open(f'ref_captions_{lang}.json', 'w', encoding='utf-8') as f:
        json.dump(ref_captions, f, indent=1, ensure_ascii=False)

