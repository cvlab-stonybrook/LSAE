import argparse
import math
import random
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

import scipy.stats as stats

try:
    import wandb
except ImportError:
    wandb = None

from model import PyramidEncoder
import torchvision.models as models
from dataset import CXR14Dataset

import pdb

class Classifier(nn.Module):
    def __init__(self, channel, out_dim, enc_state_dict=None, gray=False):
        super().__init__()
        self.encoder = PyramidEncoder(channel, gray=gray)
        self.fc = nn.Linear(2048, out_dim)
        # init encoder
        if enc_state_dict is not None:
            self.encoder.load_state_dict(enc_state_dict)
        # requires_grad(self.encoder, False)
        # self.encoder.eval()
        print("loaded")
        # init fc
        X = stats.truncnorm(-2, 2, scale=0.1)
        values = torch.as_tensor(X.rvs(self.fc.weight.numel()), dtype=self.fc.weight.dtype)
        values = values.view(self.fc.weight.size())
        with torch.no_grad():
            self.fc.weight.copy_(values)

    def forward(self, data):
        # with torch.no_grad():
        # code, _ = self.encoder(data, run_tex=False, multi_str=False)
        _, code = self.encoder(data, run_str=False,  multi_tex=False)
        # code = code.flatten(start_dim=1)
        return self.fc(code)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def bce_loss(pred, label):
    loss = F.binary_cross_entropy_with_logits(pred, label, None,
                                            pos_weight=None,
                                            reduction='mean')
    return loss

def adjust_learning_rate(optimizer, init_lr, iter, lr_steps):
    nexp = 0
    for step in lr_steps:
        if iter < step:
            break
        else:
            nexp += 1
    lr = init_lr * math.pow(10, -nexp)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_avg_auc(gt, pred, verbose=False):
    AUROCs = []
    if pred.ndim == 2:
        n_classes = pred.shape[1]
    elif pred.ndim == 1:
        n_classes = 1
    else:
        raise ValueError("Prediction shape wrong")
    for i in range(n_classes):
        try:
            auc = roc_auc_score(gt[:, i], pred[:, i])
        except (IndexError, ValueError) as error:
            if isinstance(error, IndexError):
                auc = roc_auc_score(gt_np, pred_np)
            elif isinstance(error, ValueError):
                auc = 0
            else:
                raise Exception("Unexpected Error")
        AUROCs.append(auc)
    if verbose:
        print(AUROCs)
    AUROC_avg = np.array(AUROCs).mean()
    return AUROC_avg

def train(
    args,
    tr_loader,
    ts_loader,
    classifier,
    c_optim,
    device,
):
    tr_loader = sample_data(tr_loader)

    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # define the global variables
    avg_auc = 0
    loss_dict = {}

    # create output folders
    if args.proj_name != "":
        ckpt_dir = f"outputs/ckpt-{args.proj_name}"
    else:
        ckpt_dir = "outputs/ckpt"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.distributed:
        c_module = classifier.module
    else:
        c_module = classifier

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        adjust_learning_rate(c_optim, args.lr, i, args.lr_steps)

        real_img, label = next(tr_loader)
        real_img = real_img.to(device)
        label = label.to(device)

        ##### classifier optim #####
        classifier.train()
        requires_grad(classifier, True)

        cls_logits = classifier(real_img)

        # Minimize the loss of attr classification
        cls_loss = bce_loss(cls_logits, label)

        # attr classification loss
        loss_dict["cls"] = cls_loss

        # update discriminator
        c_optim.zero_grad()
        cls_loss.backward()
        c_optim.step()

        # loss_reduced = reduce_loss_dict(loss_dict)
        cls_loss_val = loss_dict["cls"].mean().item()

        if wandb and args.wandb and i % 10 == 0:
            wandb.log(
                    {
                        "cls_loss": cls_loss_val,
                        "avg_auc": avg_auc,
                    },
                    step=i,
                )

        if i % 200 == 0:
            with torch.no_grad():
                scores_list = []
                labels_list = []
                for img, label in ts_loader:
                    img = img.to(device)
                    label = label.to(device)
                    classifier.eval()
                    cls_logits = classifier(img)
                    cls_scores = torch.sigmoid(cls_logits)
                    scores_list.append(cls_scores.cpu().data.numpy())
                    labels_list.append(label.cpu().data.numpy())
            scores_arr = np.concatenate(scores_list, axis=0)
            labels_arr = np.concatenate(labels_list, axis=0)
            if i % 1000 == 0:
                avg_auc = compute_avg_auc(labels_arr, scores_arr,verbose=True)
            else:
                avg_auc = compute_avg_auc(labels_arr, scores_arr)
        # show information
        pbar.set_description((f"c: {cls_loss_val:.4f}; avg_auc: {avg_auc:.4f}; lr: {c_optim.param_groups[0]['lr']:.4f}"))

        if i % 5000 == 0:
            torch.save(
                {
                    "c": c_module.state_dict(),
                    "c_optim": c_optim.state_dict(),
                    "args": args,
                    "avg_auc": avg_auc,
                    "iter": i,
                },
                f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            )


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, nargs="+")
    parser.add_argument("--trlist", type=str)
    parser.add_argument("--tslist", type=str)
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--mlp_r1", type=float, default=1)
    parser.add_argument("--ref_crop", type=int, default=4)
    parser.add_argument("--n_crop", type=int, default=8)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--enc_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--lr_steps", type=int, nargs="+")
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--proj_name", type=str, default="lsae_texencoder_cxr14")

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    try:
        enc_ckpt = torch.load(args.enc_ckpt, map_location=lambda storage, loc: storage)
        classifier = Classifier(args.channel, out_dim=14, enc_state_dict=enc_ckpt['e'], gray=True).to(device)
        print("Encoder dictionary loaded")
    except:
        print("Rand Init")
        classifier = Classifier(args.channel, out_dim=14, enc_state_dict=None, gray=True).to(device)
    
    classifier = classifier.to(device)
    print("Total model size: ", sum(p.numel() for p in classifier.parameters()))

    # classifier = models.resnet50(pretrained=True)
    # classifier.fc = nn.Linear(2048, 14)
    # import scipy.stats as stats
    # X = stats.truncnorm(-2, 2, scale=0.1)
    # values = torch.as_tensor(X.rvs(classifier.fc.weight.numel()), dtype=classifier.fc.weight.dtype)
    # values = values.view(classifier.fc.weight.size())
    # with torch.no_grad():
    #     classifier.fc.weight.copy_(values)
    # classifier = classifier.to(device)
    # print("Total model size: ", sum(p.numel() for p in classifier.parameters()))

    c_optim = optim.Adam(
        list(classifier.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999)
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        if args.resume:
            args.start_iter = ckpt["iter"]
            classifier.load_state_dict(ckpt["c"])
            c_optim.load_state_dict(ckpt["c_optim"])
        else:
            classifier.load_state_dict(ckpt["c"])

    path = args.path[0]

    # train dataset and dataloader
    tr_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ]
    )
    tr_dataset = CXR14Dataset(path, args.trlist, tr_transform, gray=True)
    tr_loader = data.DataLoader(
        tr_dataset,
        batch_size=args.batch,
        sampler=data_sampler(tr_dataset, shuffle=True, distributed=args.distributed),
        num_workers=8,
        drop_last=True,
    )

    # test dataset and dataloader
    ts_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ]
    )
    ts_dataset = CXR14Dataset(path, args.tslist, ts_transform, gray=True)
    ts_loader = data.DataLoader(
        ts_dataset,
        batch_size=args.batch,
        sampler=data_sampler(ts_dataset, shuffle=False, distributed=args.distributed),
        num_workers=8,
        drop_last=False,
    )

    if wandb is not None and args.wandb:
        wandb.init(project=f"proj_{args.proj_name}")


    train(
        args,
        tr_loader,
        ts_loader,
        classifier,
        c_optim,
        device,
    )
