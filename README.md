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
