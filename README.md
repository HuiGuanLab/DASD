 # Dynamic-Adapter-with-Semantics-Disentangling-for-Cross-lingual-Cross-modal-Retrieval
 This is implementation for the paper "Dynamic Adapter with Semantics Disentangling for Cross lingual Cross modal Retrieval" 
![overview](model.pdf)
 ## Table of Contents
* [Requirments](#Requirments)
* [Pretrained-models](#Pretrained-models)
* [Dataset](#Dataset)
* [Train](#Train)
* [Evaluation](#Evaluation)
# Requirments
```
torch >= 1.7.1
transformers
opencv-python
```
## Pretrained-models
The pretrained models (CLIP & M-BERT, for initialization) can be downloaded [here](https://drive.google.com/file/d/1lJU9RwuYTvEd9r9ReM9FyXRxgkxxTStx/view?usp=sharing)
```
unzip pretrained_model.zip
```


## Dataset
If you do not want the dataset and code to be placed together, please modify the 'datapath' parameter in the config file.
Download [annotations](https://drive.google.com/file/d/1LWp6RVAXUjHvljB0xUDgIg56jQRzPHcC/view?usp=sharing).
```
unzip dataset.zip
```
**Conceptual Caption images** can be crawled [here](https://ai.google.com/research/ConceptualCaptions/download). After crawled from the web, place all images under `dataset/ConceptualCaption/images`.

CC300K are used to train the released models. This subset can be found here `dataset/ConceptualCaption/cc300k.json`.

**Flickr30K images** can be [requested here](https://forms.illinois.edu/sec/229675). Untar it to `dataset/Multi30k`.
```
tar -xzvf flickr30k_images.tar.gz -C dataset/Multi30k
```
**MSCOCO images** can be downloaded and prepared with the following scripts:
```
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip

mkdir -p dataset/MSCOCO/images

unzip -d dataset/MSCOCO/images http://images.cocodataset.org/zips/train2014.zip 
unzip -d dataset/MSCOCO/images http://images.cocodataset.org/zips/val2014.zip 
unzip -d dataset/MSCOCO/images http://images.cocodataset.org/zips/test2014.zip 
```

## CCR settings:
**We conduct experiments under two CCR settings:**

(1) **Cross-lingual Finetune**: we first train models using English data in Downstream Task Dataset (DTD) and then further finetune models with target-language data produced by MT tools. Finally, models are tested on DTD target-language datasets.

(2) **Zero-shot**: models are trained on commonly-used datasets~(e.g., CC300K) and then directly evaluated on DTD without any DTD finetuning.


## Train
**Under the Cross-lingual Finetuning Setting:**
```
# Finetune on DTD English data (m30k or MSCOCO)
bash train.sh \
    expr/vitb32/Cross-lingual_Finetune/config.json 0
# For cross-lingual alignment:
bash train.sh  expr/vitb32/CLA/config.json 0
# For cross-modal alignment:
bash train.sh  expr/vitb32/CMA/config.json 0
```

**Under the Zero-shot Setting：**
```
# For cross-lingual alignment:
bash train.sh  expr/vitb32/NLT/config.json 0
# For cross-modal alignment:
bash train.sh  expr/vitb32/LE/config.json 0
```
## Evaluation
Under both settings, you can perform model evaluation after specifying the file path to the trained model in the config file:
```
bash inference.sh  expr/vitb32/LE/config.json 0
```
We release some checkpoints trained on Multi30k and MSCOCO, which can be obtained [here](https://drive.google.com/file/d/1lJU9RwuYTvEd9r9ReM9FyXRxgkxxTStx/view?usp=sharing).
