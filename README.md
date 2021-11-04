# Lung Swapping Auotencoder: Learning a Disentangled Structure-texture Representation of Chest Radiographs


This is the PyTorch implementation of [Lung Swapping Autoencoder](https://link.springer.com/chapter/10.1007%2F978-3-030-87234-2_33) published on MICCAI 2021.

<p align="center">
  <img src="https://github.com/cvlab-stonybrook/LSAE/blob/main/assets/final_pipeline.png" width="720">
</p>

## Preparation
### CXRs and Masks of ChestX-ray14
You can download our pre-processed data through the following links. Please remember to modify the data path in the command lines correspondingly.
- [Histogram Equalized ChestX-ray14 (256 x 256)](https://drive.google.com/file/d/1Mf0XI33sdhtcuvBjohe1kTmlX67-uVvz/view?usp=sharing).
- [Lung Segmentation Masks of ChestX-ray14](https://drive.google.com/file/d/1a-oH7BLrCp4ZTPembtvr3_X3ory2tH47/view?usp=sharing)
### Data Splits of ChestX-ray14
We split ChestX-ray14 following the routine of the [official website](https://nihcc.app.box.com/v/ChestXray-NIHCC). To simplify the input of annotations, we generate [train list](https://github.com/cvlab-stonybrook/LSAE/blob/main/data/trainval_list.txt) and [test list](https://github.com/cvlab-stonybrook/LSAE/blob/main/data/test_list.txt). Each line is composed of the image name and the corresponding labels like below:
```
00000001_002.png 0 1 1 0 0 0 0 0 0 0 0 0 0 0
```
If the image is positive with one class, the corresponding bit is 1, otherwise is 0. Class index follows [this](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/220660789610)

## Unsupervised Lung Swapping Pre-training
The command is following. Please fill in the blanks with your own paths.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  --master_port=8898 train_lsae.py \
  --size 256 \
  --batch 8 \
  --lr 0.001 \
  --trlist data/trainval_list.txt \
  --tslist data/test_list.txt \
  --wandb \
  --proj_name lsae \
  [CXR_PATH] [CXR_Mask_PATH]
```
We provide the trained [LSAE checkpoint](https://drive.google.com/file/d/1Qh-BhnAQIdnvO7bd--RArIOtzobwR9FQ/view?usp=sharing) which is used to perform the downstream tasks.

### Qualitative Results of Lung Swapping
<p align="center">
  <img src="https://github.com/cvlab-stonybrook/LSAE/blob/main/assets/final_teaser.png" width="720">
</p>

## Full Labeled Data Finetuning on ChestX-ray14
The command is following. Please fill in the blanks with your own paths. Before running, you need to download the [pretrained_lsae.pt](https://drive.google.com/file/d/1Qh-BhnAQIdnvO7bd--RArIOtzobwR9FQ/view?usp=sharing), and put it in the directory *saved_ckpts*.
```
CUDA_VISIBLE_DEVICES=0 python train_texencoder_cxr14.py \
  --path [CXR_PATH] \
  --batch 96 \
  --iter 35000 \
  --lr 0.01 \
  --lr_steps 26000 30000 \
  --trlist data/trainval_list.txt \
  --tslist data/test_list.txt \
  --enc_ckpt saved_ckpts/pretrained_lsae.pt \
  --wandb
```
<img src="https://latex.codecogs.com/gif.latex?Enc^t" title="Enc^t" /> in LSAE achieves 79.2%(mAUC) on ChestX-ray14. The quantitative comparison with Inception v3 and DenseNet 121 is shown in the following table, together with all the model weights.
| Models | Init Weights | Params(M) | mAUC(%) |
| :----: | :----------: | :-------: | :-----: |
| DenseNet 121 [[ckpt](https://drive.google.com/file/d/1HIOoprsTtWB_-rKNxzzx4HOk3qhlPgiM/view?usp=sharing)] | ImageNet pre-trained | 7 | 78.7 |
| Inception v3 [[ckpt](https://drive.google.com/file/d/1O5RNYo4C-i33BIGR73IuzEagdmCPL69k/view?usp=sharing)] | ImageNet pre-trained | 22 | 79.6 |
| <img src="https://latex.codecogs.com/gif.latex?Enc^t" title="Enc^t" /> in LSAE [[ckpt](https://drive.google.com/file/d/1lYAofe93BvvhYaxICDS3RLOItqwe7Qs4/view?usp=sharing)] | LSAE pre-trained | 5 | 79.2 |

## Cite
```
@inproceedings{zhou2021chest,
  title={Chest Radiograph Disentanglement for COVID-19 Outcome Prediction},
  author={Zhou, Lei and Bae, Joseph and Liu, Huidong and Singh, Gagandeep and Green, Jeremy and Samaras, Dimitris and Prasanna, Prateek},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={345--355},
  year={2021},
  organization={Springer}
}
```

## Acknowledgement
Our code is heavily based on the following open-sourced repositories. We appreciate their generous releases.
- [Swapping Autoencoder](https://github.com/rosinality/swapping-autoencoder-pytorch) by [rosinality](https://github.com/rosinality)
- [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) by [rosinality](https://github.com/rosinality)
- [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation) by [taesungp](https://github.com/taesungp)
- [WarpAffine2GridSample](https://github.com/wuneng/WarpAffine2GridSample) by [wuneng](https://github.com/wuneng)
