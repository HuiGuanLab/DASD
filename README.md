 # Dynamic Adapter with Semantics Disentangling for Cross-Lingual Cross-Modal Retrieval
This is the implementation for the paper "Dynamic Adapter with Semantics Disentangling for Cross-lingual Cross-modal Retrieval" 

 ![](https://github.com/zhiyudongg/DASD/blob/main/framework.png)
 
 ## Table of Contents
* [Requirments](#Requirments)
* [Pretrained models used in DASD](#Pretrained-models-used-in-our-DASD)
* [Datasets](#Datasets)
* [CCR settings](#CCR-settings)
* [Training](#Training)
* [Evaluation](#Evaluation)
## Requirments
```shell script
torch >= 1.7.1
transformers
opencv-python
```
## Pretrained models used in our DASD
The pretrained models used in DASD (CLIP & mBERT, for initialization) can be downloaded [here](https://drive.google.com/file/d/1lJU9RwuYTvEd9r9ReM9FyXRxgkxxTStx/view?usp=sharing):
```shell script
unzip pretrained_model.zip
```


## Datasets
If you do not want the dataset and code to be placed together, please modify the 'datapath' parameter in the configuration file.

Download [captions](https://drive.google.com/file/d/1rbVW71UrnyRKNVe2Y3WeNJuoJytG_kaO/view?usp=sharing) used in our experiments and unzip it to `./dataset/`:
```shell script
unzip dataset.zip
```
**Conceptual Caption images** can be crawled [here](https://ai.google.com/research/ConceptualCaptions/download). After crawled from the web, place all images under `dataset/ConceptualCaption/images`.

**CC300K**  are also used to train the released models. This subset can be found here `dataset/ConceptualCaption/cc300k.json`.

**Flickr30K images** can be [requested here](https://forms.illinois.edu/sec/229675). Untar it to `dataset/Multi30k`.
```shell script
tar -xzvf flickr30k_images.tar.gz -C dataset/Multi30k
```
**MSCOCO images** can be downloaded and prepared with the following scripts:
```shell script
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip

mkdir -p dataset/MSCOCO/images

unzip -d dataset/MSCOCO/images http://images.cocodataset.org/zips/train2014.zip 
unzip -d dataset/MSCOCO/images http://images.cocodataset.org/zips/val2014.zip 
unzip -d dataset/MSCOCO/images http://images.cocodataset.org/zips/test2014.zip 
```

## CCR settings
**We conduct experiments under two CCR settings:**

(1) **Cross-lingual Finetune**: we first train models using English data in Downstream Task Dataset (DTD) and then further finetune models with target-language data produced by MT tools. Finally, models are tested on DTD target-language datasets.

(2) **Zero-shot**: models are trained on commonly-used datasets~(e.g., CC300K) and then directly evaluated on DTD without any DTD finetuning.



## Training
**Under the Cross-lingual Finetuning Setting,**
we train the model using the following scripits:
```shell script
# Finetune on DTD English data:
bash train.sh  expr/vitb32/Cross-lingual_Finetune/config.json 0

# For cross-lingual cross-modal alignment:
bash CLCMA.sh 0
```

**Under the Zero-shot Setting,**
we train the model using the following scripits:
```shell script
# For cross-lingual cross-modal alignment:
bash CLCMA.sh 0
```
For both settings, please specify the training dataset in the corresponding configuration file (config.json).

## Evaluation
For both settings, we use the same command for evalution: 
```shell script
bash inference.sh  expr/vitb32/CMA/config.json 0
```
You can specify the test dataset and trained model in the corresponding configuration file (config.json).

We release some checkpoints trained on Multi30k and MSCOCO, which can be obtained [here](https://drive.google.com/file/d/1oDkX7-4enhU83fjkHkP2R7pwEG9cRKpH/view?usp=sharing).
