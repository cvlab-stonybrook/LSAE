import argparse
import math
import random
import os
import cv2
from functools import partial

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from model import PyramidEncoder, Generator, Discriminator, Cooccurv2Discriminator, MultiProjectors, PatchNCELoss
from dataset import CXR14maskDataset
from stylegan2.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)

def cvt_MToTheta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`
    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required
    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]

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


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real_list = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real_list[0].pow(2).reshape(grad_real_list[0].shape[0], -1).sum(1).mean()
    if len(grad_real_list) > 1:
        for grad_real in grad_real_list[1:]:
            grad_penalty += grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def differential_crop_w_random_rotation(image, max_angle):
    """
    max_angle defines the maximum rotation angles from [-max_angle, max_angle]
    """
    thetas = []
    N, D, H, W = image.shape
    for i in range(N):
        angle = random.uniform(-max_angle, max_angle)
        # angle = 60
        center = (W//2, H//2)
        scale = math.cos(math.pi*abs(angle)/180) + math.sin(math.pi*abs(angle)/180)*H/W
        # scale = 1.6
        # print(scale)
        affine_trans = cv2.getRotationMatrix2D(center, angle, scale)
        theta = cvt_MToTheta(affine_trans, W, H)
        thetas.append(torch.from_numpy(theta))
    thetas = torch.stack(thetas).to(image.device)
    grid = F.affine_grid(thetas, image.size(), align_corners=False)
    rotated = F.grid_sample(image, grid.float(), align_corners=False)

    return rotated

def raw_patchify_image(img, n_crop, mask=None, min_size=1 / 16, max_size=1 / 8, max_angle=60):
    batch, channel, height, width = img.shape

    def compute_corner(cent_x, cent_y, c_w, c_h):
        c_x = cent_x - c_w//2
        c_y = cent_y - c_h//2
        return c_x, c_y

    default_mask = torch.zeros(batch, height, width)
    default_mask[:, height//4:3*height//4, width//4:3*width//4] = 1

    if mask is None:
        mask = default_mask
    mask = mask.squeeze()

    crop_size = torch.rand(batch, n_crop) * (max_size - min_size) + min_size
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()
    crop_w = (crop_size * width).type(torch.int64).tolist()
    min_h = int(height * min_size)
    min_w = int(width * min_size)

    patches = []
    for b in range(batch):
        indices = torch.nonzero(mask[b]) # [height, width]
        if indices.size(0) == 0:
            indices = torch.nonzero(default_mask[b])
        for _ in range(n_crop):
            # random sample origin
            ind = random.randrange(0, indices.size(0))
            cent_y, cent_x = indices[ind].tolist()
            # random sample crop size
            crop_ratio = random.uniform(0,1) * (max_size - min_size) + min_size
            c_h, c_w = int(height * crop_ratio), int(width * crop_ratio)
            # recompute corners and area
            c_x, c_y = compute_corner(cent_x, cent_y, c_w, c_h)
            # clip the coordinates
            if c_y < 0:
                c_y = 0
            if c_x < 0:
                c_x = 0
            if c_y + c_h >= height:
                c_y = height - c_h - 1
            if c_x + c_w >= width:
                c_x = width - c_w - 1

            init_patch = img[b, :, c_y : c_y + c_h, c_x : c_x + c_w].view(1, channel, c_h, c_w)
            intp_patch = F.interpolate(init_patch, size=(target_h, target_w), mode="bilinear", align_corners=False)
            patches.append(intp_patch)
    # patches shape: [batch*n_crop, channel, target_h, target_w]
    patches = torch.cat(patches, dim=0)
    rotated = differential_crop_w_random_rotation(patches, max_angle)

    return rotated

def sample_patches(feat_list, n_crop, mask=None, coords=None, inv=False):
    if inv and mask is not None:
        mask = 1 - mask
    if mask is not None:
        mask = mask.squeeze()


    # collect info
    batchSize = feat_list[0].size(0)
    channels = []
    spt_dims = []
    for feat in feat_list:
        assert(feat.size(0) == batchSize), "Batch size of features should be consistent"
        channels.append(feat.size(1))
        spt_dims.append(feat.shape[2:])

    if coords is None:
        # sample coords from mask
        batch_coords = []
        for b in range(batchSize):
            # height and width of mask
            h, w = mask[b].shape
            # get valid candidates from mask
            indices = torch.nonzero(mask[b])
            if indices.size(0) == 0:
                indices = torch.nonzero(torch.ones(h, w))
            # sample points
            normed_coords = []
            for _ in range(n_crop):
                ind = random.randrange(0, indices.size(0))
                cent_y, cent_x = indices[ind].tolist()
                normed_coords.append((cent_y / h, cent_x / w))
            batch_coords.append(normed_coords)
    else:
        batch_coords = coords

    # extract features according to sampled points
    scale_feats = []
    for i, feat in enumerate(feat_list):
        h, w = spt_dims[i]
        sampled_feats = []
        for b in range(batchSize):
            for j in range(n_crop):
                cent_y, cent_x = min(h-1, int(batch_coords[b][j][0] * h)), min(w-1, int(batch_coords[b][j][1] * w))
                sampled_feats.append(feat[b, :, cent_y, cent_x])
        # [batchSize*n_crop, channel]
        sampled_feats = torch.stack(sampled_feats, dim=0)
        scale_feats.append(sampled_feats)

    return scale_feats, batch_coords

def train(
    args,
    loaders,
    encoder,
    generator,
    str_projectors,
    discriminator,
    cooccur,
    g_optim,
    d_optim,
    e_ema,
    g_ema,
    device,
):
    patchify_image = partial(raw_patchify_image, min_size=args.min_patch, max_size=args.max_patch)

    loader = sample_data(loaders[0])
    ts_loader = sample_data(loaders[1])

    patchnce_loss = PatchNCELoss(nce_T=0.07, batch=args.batch//2)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    cooccur_r1_loss = torch.tensor(0.0, device=device)
    feat_recon_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    loss_dict = {}

    # create output folders
    if args.proj_name != "":
        sample_dir = f"outputs/sample-{args.proj_name}"
        ckpt_dir = f"outputs/ckpt-{args.proj_name}"
    else:
        sample_dir = "outputs/sample"
        ckpt_dir = "outputs/ckpt"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.distributed:
        e_module = encoder.module
        g_module = generator.module
        s_module = str_projectors.module
        d_module = discriminator.module
        c_module = cooccur.module

    else:
        e_module = encoder
        g_module = generator
        s_module = str_projectors
        d_module = discriminator
        c_module = cooccur

    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img, real_msk, _ = next(loader)
        real_img = real_img.to(device)

        requires_grad(encoder, False)
        requires_grad(generator, False)
        requires_grad(str_projectors, False)
        requires_grad(discriminator, True)
        requires_grad(cooccur, True)

        real_img1, real_img2 = real_img.chunk(2, dim=0)
        real_msk1, real_msk2 = real_msk.chunk(2, dim=0)

        structure1, texture1 = encoder(real_img1, multi_tex=False)
        _, texture2 = encoder(real_img2, run_str=False, multi_tex=False)

        # image adversarial loss
        fake_img1 = generator(structure1, texture1)
        fake_img2 = generator(structure1, texture2)
        fake_pred = discriminator(torch.cat((fake_img1, fake_img2), 0))
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        # texture adversarial loss
        fake_patch = patchify_image(fake_img2, args.n_crop, mask=real_msk1)
        real_patch = patchify_image(real_img2, args.n_crop, mask=real_msk2)
        ref_patch = patchify_image(real_img2, args.ref_crop * args.n_crop, mask=real_msk2)
        fake_patch_pred, ref_input = cooccur(
            fake_patch, args.n_crop, reference=ref_patch, ref_batch=args.ref_crop
        )
        real_patch_pred, _ = cooccur(real_patch, args.n_crop, ref_input=ref_input)
        cooccur_loss = d_logistic_loss(real_patch_pred, fake_patch_pred)

        loss_dict["d"] = d_loss
        loss_dict["cooccur"] = cooccur_loss

        d_optim.zero_grad()
        (d_loss + cooccur_loss).backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            real_patch.requires_grad = True
            real_patch_pred, _ = cooccur(real_patch, args.n_crop, reference=ref_patch, ref_batch=args.ref_crop)
            cooccur_r1_loss = d_r1_loss(real_patch_pred, real_patch)

            d_optim.zero_grad()

            r1_loss_sum = args.r1 / 2 * r1_loss * args.d_reg_every
            r1_loss_sum += args.cooccur_r1 / 2 * cooccur_r1_loss * args.d_reg_every
            r1_loss_sum += 0 * real_pred[0, 0] + 0 * real_patch_pred[0, 0]
            r1_loss_sum.backward()

            d_optim.step()
            # real_img.requires_grad = False

        loss_dict["r1"] = r1_loss
        loss_dict["cooccur_r1"] = cooccur_r1_loss

        requires_grad(encoder, True)
        requires_grad(generator, True)
        requires_grad(str_projectors, True)
        requires_grad(discriminator, False)
        requires_grad(cooccur, False)

        structure1_list, texture1 = encoder(real_img1, multi_str=True, multi_tex=False)
        _, texture2 = encoder(real_img2, run_str=False, multi_tex=False)
        structure1 = structure1_list[-1]

        fake_img1 = generator(structure1, texture1)
        fake_img2 = generator(structure1, texture2)

        # reconstruction loss
        recon_loss = F.l1_loss(fake_img1, real_img1.detach())

        # image adversarial loss
        fake_pred = discriminator(torch.cat((fake_img1, fake_img2), 0))
        g_loss = g_nonsaturating_loss(fake_pred)

        # texture adversarial loss
        fake_patch = patchify_image(fake_img2, args.n_crop, mask=real_msk1)
        ref_patch = patchify_image(real_img2, args.ref_crop * args.n_crop, mask=real_msk2)
        fake_patch_pred, _ = cooccur(fake_patch, args.n_crop, reference=ref_patch, ref_batch=args.ref_crop)
        g_cooccur_loss = g_nonsaturating_loss(fake_patch_pred)

        # Patch NCE loss
        # re-encode
        fake_structure1_list, fake_texture2 = encoder(fake_img2, multi_str=True, multi_tex=False)
        fake_patch_vectors, coords = sample_patches(fake_structure1_list[:-1], args.n_crop, mask=real_msk1, inv=True)
        real_patch_vectors, _ = sample_patches(structure1_list[:-1], args.n_crop, coords=coords)
        str_qs = str_projectors(fake_patch_vectors)
        str_ks = str_projectors(real_patch_vectors)
        ## compute loss
        g_pnce_loss = 0
        num_scales = len(str_qs)
        for str_q, str_k in zip(str_qs, str_ks):
            g_pnce_loss += patchnce_loss(str_q, str_k) / num_scales
            # g_pnce_loss += patchnce_loss(str_k, str_k)

        # feature reconstruction loss
        feat_recon_loss = 0.5 * F.mse_loss(fake_structure1_list[-1], structure1.detach()) + 0.5 * F.mse_loss(fake_texture2, texture2.detach())

        loss_dict["recon"] = recon_loss
        loss_dict["g"] = g_loss
        loss_dict["g_cooccur"] = g_cooccur_loss
        loss_dict["g_pnce"] = g_pnce_loss
        loss_dict["feat_recon"] = feat_recon_loss


        g_optim.zero_grad()
        (recon_loss + 0.5*g_loss + g_pnce_loss + g_cooccur_loss + 0.5*feat_recon_loss).backward() #  + 10*
        g_optim.step()

        accumulate(e_ema, e_module, accum)
        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        cooccur_val = loss_reduced["cooccur"].mean().item()
        recon_val = loss_reduced["recon"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        g_cooccur_val = loss_reduced["g_cooccur"].mean().item()
        g_pnce_val = loss_reduced["g_pnce"].mean().item()
        feat_recon_val = loss_reduced["feat_recon"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        cooccur_r1_val = loss_reduced["cooccur_r1"].mean().item()


        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; c: {cooccur_val:.4f} g: {g_loss_val:.4f}; g_cooccur: {g_cooccur_val:.4f}; "
                    f"recon: {recon_val:.4f}; feat_recon: {feat_recon_val:.4f}; pnce: {g_pnce_val:.4f}; "
                    f"r1: {r1_val:.4f}; r1_cooccur: {cooccur_r1_val:.4f}"
                )
            )

            if wandb and args.wandb and i % 10 == 0:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Cooccur": cooccur_val,
                        "Recon": recon_val,
                        "Feat Recon": feat_recon_val,
                        "Generator Cooccur": g_cooccur_val,
                        "R1": r1_val,
                        "Cooccur R1": cooccur_r1_val,
                        "PNCE": g_pnce_val,
                    },
                    step=i,
                )

            if i % 200 == 0:
                with torch.no_grad():
                    # read test image
                    real_img, _, _ = next(ts_loader)
                    real_img = real_img.to(device)
                    real_img1, real_img2 = real_img.chunk(2, dim=0)

                    e_ema.eval()
                    g_ema.eval()

                    structure1, texture1 = e_ema(real_img1, multi_tex=False)
                    _, texture2 = e_ema(real_img2, run_str=False, multi_tex=False)

                    fake_img1 = g_ema(structure1, texture1)
                    fake_img2 = g_ema(structure1, texture2)

                    sample = torch.cat((real_img1, fake_img1, fake_img2, real_img2), 0)

                    utils.save_image(
                        sample,
                        f"{sample_dir}/{str(i).zfill(6)}.png",
                        nrow=sample.shape[0] // 4,
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "g": g_module.state_dict(),
                        "s": s_module.state_dict(),
                        "d": d_module.state_dict(),
                        "cooccur": c_module.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"{ckpt_dir}/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, nargs="+")
    parser.add_argument("--trlist", type=str)
    parser.add_argument("--tslist", type=str)
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--min_patch", type=float, default=1/16)
    parser.add_argument("--max_patch", type=float, default=1/4)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--cooccur_r1", type=float, default=1)
    parser.add_argument("--ref_crop", type=int, default=4)
    parser.add_argument("--n_crop", type=int, default=8)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    # a new param, weights of scales
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--proj_name", type=str, default="")

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    encoder = PyramidEncoder(args.channel, gray=True).to(device)
    generator = Generator(args.channel, gray=True).to(device)
    str_projectors = MultiProjectors([args.channel, args.channel * 2, args.channel * 8], use_mlp=True).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, gray=True
    ).to(device)
    cooccur = Cooccurv2Discriminator(args.channel, size=args.size*args.max_patch, gray=True).to(device)

    e_ema = PyramidEncoder(args.channel, gray=True).to(device)
    g_ema = Generator(args.channel, gray=True).to(device)
    e_ema.eval()
    g_ema.eval()
    accumulate(e_ema, encoder, 0)
    accumulate(g_ema, generator, 0)

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()) + list(str_projectors.parameters()),
        lr=args.lr,
        betas=(0, 0.99),
    )
    d_optim = optim.Adam(
        list(discriminator.parameters()) + list(cooccur.parameters()),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        encoder.load_state_dict(ckpt["e"])
        generator.load_state_dict(ckpt["g"])
        str_projectors.load_state_dict(ckpt['s'])
        discriminator.load_state_dict(ckpt["d"])
        cooccur.load_state_dict(ckpt["cooccur"])
        e_ema.load_state_dict(ckpt["e_ema"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        str_projectors = nn.parallel.DistributedDataParallel(
            str_projectors,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        cooccur = nn.parallel.DistributedDataParallel(
            cooccur,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ]
    )

    dataset = CXR14maskDataset(args.path[0], args.path[1], args.trlist, transform, gray=True)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=4,
    )

    # test dataset
    ts_dataset = CXR14maskDataset(args.path[0], args.path[1], args.tslist, transform, gray=True)
    ts_loader = data.DataLoader(
        ts_dataset,
        batch_size=args.batch,
        sampler=data_sampler(ts_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=f"proj_{args.proj_name}")

    train(
        args,
        [loader, ts_loader],
        encoder,
        generator,
        str_projectors,
        discriminator,
        cooccur,
        g_optim,
        d_optim,
        e_ema,
        g_ema,
        device,
    )
