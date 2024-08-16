 # Dynamic-Adapter-with-Semantics-Disentangling-for-Cross-lingual-Cross-modal-Retrieval
# Requirments
```
torch >= 1.7.1
transformers
opencv-python
```
## Pretrained models
The pretrained models (CLIP & M-BERT, for initialization) can be downloaded [here](https://drive.google.com/file/d/1lJU9RwuYTvEd9r9ReM9FyXRxgkxxTStx/view?usp=sharing)
```
unzip pretrained_model.zip
```


## Data preparation
If you do not want the dataset and code to be placed together, please modify the 'datapath' parameter in the config file.
Download [annotations](https://drive.google.com/file/d/1LWp6RVAXUjHvljB0xUDgIg56jQRzPHcC/view?usp=sharing) and unzip it to `./dataset/`

**Conceptual Caption images** can be crawled [here](https://ai.google.com/research/ConceptualCaptions/download). After crawled from the web, place all images under `dataset/ConceptualCaption/images`

CC300K are used to train the released models. This subset can be found here `dataset/ConceptualCaption/cc300k.json`

**Flickr30K images** can be [requested here](https://forms.illinois.edu/sec/229675). Untar it to `dataset/Multi30k`
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

## Train
```
# NLT stage
bash train.sh  expr/vitb32/NLT/config.json 0
# LE stage:
bash train.sh  expr/vitb32/LE/config.json 0
```
**Finetune on DTD(m30k or MSCOCO)**
```
bash train.sh \
    expr/vitb32/Cross-lingual_Finetune/config.json 0
```

