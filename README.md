# iMMG

This is the implementation of our paper "Ingestible magnetomotiliography (iMMG) for continuous monitoring and modulation of intestinal motility", which is currently under review.

# Hardware requirements
We recommend running the code on a single RTX 4090 GPU.

# Installation
First, create a conda virtual environment and activate it:
```
conda create -n iMMG python=3.9 -y
conda activate iMMG
```

Then, install the following packages:

- [Pytorch with cuda](https://pytorch.org): `conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia`
- numpy: ```pip install numpy==1.26.4```
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install fvcore==0.1.5.post20221221`
- simplejson: ```pip install simplejson==3.19.2```
- einops: ```pip install einops==0.7.0```

Finally, build the iMMG codebase by running:

```
git clone https://github.com/hudelin24/iMMG
cd iMMG
```
# Usage
## Data Preparation

Download our data via [link]([https://mycuhk-my.sharepoint.com/:u:/g/personal/1155186674_link_cuhk_edu_hk/Ed5wwtiyxJdGuGdrdhhUWtsBqrPG8XYyEwkMxBEKeYRHIQ?e=3jAD7n](https://gocuhk-my.sharepoint.com/:u:/g/personal/delinhu_cuhk_edu_hk/IQCxvLdKuAaST7kVEduqiA5WAfNlAYcJgvaSftFk4XXdhZk?e=UjHpZR) or from Zenodo (release soon) to `iMMG` folder and uncompress.

## Training
```
python Magconv/tools/run_mcnn.py \
  --cfg Magconv/configs/MCNN_train.yaml \
  GPU_ENABLE True \
  GPU_ID 0 \
  DATA.PATH_TO_DATA_DIR Data/ \
  OUTPUT_DIR Magconv/results/swine  
```

## Testing
```
python Magconv/tools/run_mcnn.py \
  --cfg Magconv/configs/MCNN_train.yaml \
  GPU_ENABLE True \
  GPU_ID 0 \
  TRAIN.ENABLE False \ 
  TEST.CHECKPOINT_FILE_PATH: Magconv/trained_NNs/CNN_swine/checkpoint_epoch_00020.pyth \
  DATA.PATH_TO_DATA_DIR Data/ \
  OUTPUT_DIR Magconv/results/swine  
```






