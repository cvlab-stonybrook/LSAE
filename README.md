## Lung Swapping Auotencoder: Learning a Disentangled Structure-texture Representation of Chest Radiographs

*stay tuned. More to upload*

This is the official code for Lung Swapping Autoencoder published on MICCAI 2021

<p align="center">
  <img src="https://github.com/cvlab-stonybrook/LSAE/tree/main/assets/final_teaser.pdf" width="300">
</p>

### Preparation
#### Packages
Install packages
#### Data
- [Histogram Equalized ChestX-ray14(256 x 256)](https://drive.google.com/file/d/1Mf0XI33sdhtcuvBjohe1kTmlX67-uVvz/view?usp=sharing)
- [Lung Segmentation Masks of ChestX-ray14](https://drive.google.com/file/d/1a-oH7BLrCp4ZTPembtvr3_X3ory2tH47/view?usp=sharing)
- train_val_list.txt *to be uploaded*
- test_list.txt *to be uploaded*

### Unsupervised Lung Swapping Pre-training
The command is following. Please fill in the blanks with your own paths.
```
bash scripts/train_lsae.sh
```

### Full Labeled Data Finetuning on ChestX-ray14
The command is following. Please fill in the blanks with your own paths.

```
bash scripts/train_texencoder_cxr14.sh
```