# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import logging
import math
import os
import random
import shutil
import sys

from collections import defaultdict
from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pytorch_msssim import ms_ssim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset_utils.scv_dataset import SCVFolder
from dataset_utils.video_dataset import VideoFolder


# from compressai.datasets import VideoFolder
# from compressai.utils.bench.codecs import compute_metrics
from compressai.zoo import video_models
from examples.codec import torch2img



from src.models.video_net_dmc import DMC
from utils import util

def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m
def collect_likelihoods_list(likelihoods_list, num_pixels: int):
    bpp_info_dict = defaultdict(int)
    bpp_loss = 0

    for i, frame_likelihoods in enumerate(likelihoods_list):
        frame_bpp = 0
        for label, likelihoods in frame_likelihoods.items():
            label_bpp = 0
            for field, v in likelihoods.items():
                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)

                bpp_loss += bpp
                frame_bpp += bpp
                label_bpp += bpp

                bpp_info_dict[f"bpp_loss.{label}"] += bpp.sum()
                bpp_info_dict[f"bpp_loss.{label}.{i}.{field}"] = bpp.sum()
            bpp_info_dict[f"bpp_loss.{label}.{i}"] = label_bpp.sum()
        bpp_info_dict[f"bpp_loss.{i}"] = frame_bpp.sum()
    return bpp_loss, bpp_info_dict


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, return_details: bool = False, bitdepth: int = 8, metrics = 'mse'):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scaling_functions = lambda x: x #lambda x: (2**bitdepth - 1) ** 2 * x
        self.return_details = bool(return_details)

        self.metrics = metrics
    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for frame_likelihoods in likelihoods_list
            for likelihoods in frame_likelihoods
        )

    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f"len(x)={len(x)} != len(target)={len(target)})")

        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError(
                "number of channels mismatches while computing distortion"
            )

        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)

        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)

        # compute metric over each component (eg: y, u and v)
        metric_values = []
        for (x0, x1) in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
            isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst):
        if (
            not isinstance(lst, (tuple, list))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError(
                "Expected a list of 4D torch.Tensor (or tuples of) as input"
            )

    def forward(self, output, target):
        assert isinstance(target, type(output["x_hat"]))
        assert len(output["x_hat"]) == len(target)

        self._check_tensors_list(target)
        self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W * num_frames

        # Get scaled and raw loss distortions for each frame
        scaled_distortions = []
        distortions = []
        if self.metrics == 'ms-ssim':
            ms_ssim_distortions = []
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)

            distortions.append(distortion)
            scaled_distortions.append(scaled_distortion)
            if self.metrics == 'ms-ssim':
                ms_ssim_distortion = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
                ms_ssim_distortions.append(ms_ssim_distortion)
            if self.return_details:
                out[f"frame{i}.mse_loss"] = distortion
        # aggregate (over batch and frame dimensions).
        out["mse_loss"] = torch.stack(distortions).mean()
        if self.metrics == 'ms-ssim':
            out["ms_ssim_loss"] = torch.stack(ms_ssim_distortions).mean()
        else:
            out["ms_ssim_loss"] = None
        # average scaled_distortions accros the frames
        scaled_distortions = sum(scaled_distortions) / num_frames

        assert isinstance(output["likelihoods"], list)
        likelihoods_list = output.pop("likelihoods")

        # collect bpp info on noisy tensors (estimated differentiable entropy)
        bpp_loss, bpp_info_dict = collect_likelihoods_list(likelihoods_list, num_pixels)
        if self.return_details:
            out.update(bpp_info_dict)  # detailed bpp: per frame, per latent, etc...

        # now we either use a fixed lambda or try to balance between 2 lambdas
        # based on a target bpp.
        lambdas = torch.full_like(bpp_loss, self.lmbda)

        bpp_loss = bpp_loss.mean()
        out["loss"] = (lambdas * scaled_distortions).mean() + bpp_loss

        out["distortion"] = scaled_distortions.mean()
        out["bpp_loss"] = bpp_loss
        if self.metrics == 'ms-ssim':
            out["loss"] = (lambdas * out["ms_ssim_loss"]).mean() + bpp_loss
        return out

class StageRateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, return_details: bool = False, bitdepth: int = 8, metrics = 'mse'):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scaling_functions = lambda x: x #(2**bitdepth - 1) ** 2 * x
        # self._scaling_functions = lambda x:  x
        self.return_details = bool(return_details)

        self.metrics = metrics
    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for frame_likelihoods in likelihoods_list
            for likelihoods in frame_likelihoods
        )

    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f"len(x)={len(x)} != len(target)={len(target)})")

        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError(
                "number of channels mismatches while computing distortion"
            )

        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)

        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)

        # compute metric over each component (eg: y, u and v)
        metric_values = []
        for (x0, x1) in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
            isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst):
        if (
            not isinstance(lst, (tuple, list))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError(
                "Expected a list of 4D torch.Tensor (or tuples of) as input"
            )

    def forward(self, output, target, global_step):

        if global_step < 200000:
            motionmse = 1
            motionbpp = 1
            residualbpp = 0
            finalmse = 0
        elif global_step < 400000:
            # self.TrainwoMotion()
            motionmse = 0
            motionbpp = 0
            residualbpp = 0
            finalmse = 1
        elif global_step < 500000:
            # TrainwoMotion()
            motionmse = 0
            motionbpp = 0
            residualbpp = 1
            finalmse = 1
        else:
            # Trainall()
            motionmse = 0
            motionbpp = 1
            residualbpp = 1
            finalmse = 1
        assert isinstance(target, type(output["x_hat"]))
        assert len(output["x_hat"]) == len(target)

        self._check_tensors_list(target)
        self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W * num_frames

        # Get scaled and raw loss distortions for each frame
        scaled_distortions = []
        distortions = []
        scaled_distortions_motion = []
        distortions_motion = []
        if self.metrics == 'ms-ssim':
            ms_ssim_distortions = []
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)
            scaled_distortion_motion, distortion_motion = self._get_scaled_distortion(output["x_pred"][i], x)
            distortions.append(distortion)
            scaled_distortions.append(scaled_distortion)
            distortions_motion.append(distortion_motion)
            scaled_distortions_motion.append(scaled_distortion_motion)
            if self.metrics == 'ms-ssim':
                ms_ssim_distortion = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
                ms_ssim_distortions.append(ms_ssim_distortion)
            if self.return_details:
                out[f"frame{i}.mse_loss"] = distortion
                out[f"frame{i}.warp_loss"] = distortion_motion
        # aggregate (over batch and frame dimensions).
        out["mse_loss"] = torch.stack(distortions).mean()
        out["motion_warp_loss"] = torch.stack(distortions_motion).mean()
        if self.metrics == 'ms-ssim':
            out["ms_ssim_loss"] = torch.stack(ms_ssim_distortions).mean()
        else:
            out["ms_ssim_loss"] = None
        # average scaled_distortions accros the frames
        scaled_distortions = sum(scaled_distortions) / num_frames

        assert isinstance(output["likelihoods"], list)
        likelihoods_list = output.pop("likelihoods")

        # collect bpp info on noisy tensors (estimated differentiable entropy)
        bpp_loss, bpp_info_dict = collect_likelihoods_list(likelihoods_list, num_pixels)
        if self.return_details:
            out.update(bpp_info_dict)  # detailed bpp: per frame, per latent, etc...

        # now we either use a fixed lambda or try to balance between 2 lambdas
        # based on a target bpp.
        lambdas = torch.full_like(bpp_loss, self.lmbda)

        bpp_loss = bpp_loss.mean()
        # out["loss"] = (lambdas * scaled_distortions).mean() + bpp_loss
        out["loss"] = (lambdas * (finalmse * scaled_distortions + motionmse * out["motion_warp_loss"])).mean() + motionbpp * out["bpp_loss.motion"] + residualbpp * out["bpp_loss.residual"]
        out["distortion"] = scaled_distortions.mean()
        out["bpp_loss"] = bpp_loss
        if self.metrics == 'ms-ssim':
            out["loss"] = (lambdas * out["ms_ssim_loss"]).mean() + bpp_loss
        return out
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer


def train_one_epoch(
    model, train_dataloader, optimizer, epoch, clip_max_norm, logger_train, logger_total, tb_logger, current_step, args
):
    model.train()
    device = next(model.parameters()).device
    if args.multi_frames > 2:
        motionmse, motionbpp, residualbpp, finalmse = model.SingleTrainstage(epoch)
    else:
        motionmse, motionbpp, residualbpp, finalmse = model.MultiTrainstage(epoch)

    for i, batch in enumerate(train_dataloader):
        d = [frames.to(device) for frames in batch]
        # print(i)
        optimizer.zero_grad()

        # if current_step % 10000 == 0:

        out_net = model(d)

        # out_criterion = criterion(out_net, d[1:], current_step)
        # out_criterion["loss"].backward()

        mse_loss = out_net['mse_loss']
        warploss = out_net['warploss']

        bpp_y = out_net["bpp_y"]
        bpp_z = out_net["bpp_z"]
        bpp_mv_y = out_net["bpp_mv_y"]
        bpp_mv_z = out_net["bpp_mv_z"]
        bpp = out_net["bpp"]
        if args.finetune:
            mse_loss = out_net['weighted_mse_loss']
        distortion = finalmse * mse_loss + motionmse * warploss
        bpp_loss = motionbpp * (bpp_mv_y+bpp_mv_z) + residualbpp * (bpp_y+bpp_z)
        # rd_loss = train_lambda * distortion + distribution_loss
        rd_loss = args.lmbda * distortion + bpp_loss
        # tc = datetime.datetime.now()
        rd_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        # if i % 10 == 0:
        #     print(
        #         f"Train epoch {epoch}: ["
        #         f"{i*len(d)}/{len(train_dataloader.dataset)}"
        #         f" ({100. * i / len(train_dataloader):.0f}%)]"
        #         f'\tLoss: {out_criterion["loss"].item():.3f} |'
        #         f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
        #         f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'

        #     )
        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), rd_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), bpp_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_motion'), (bpp_mv_y+bpp_mv_z).item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_residual'), (bpp_y+bpp_z).item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: mse_loss'), mse_loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: warp_loss'), warploss.item(), current_step)
            # if out_criterion["mse_loss"] is not None:
            #     tb_logger.add_scalar('{}'.format('[train]: mse_loss'), mse_loss.item(), current_step)
            #     tb_logger.add_scalar('{}'.format('[train]: warp_loss'), warploss.item(), current_step)
            # if out_criterion["ms_ssim_loss"] is not None:
            #     tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)
        if current_step % 50000==0:
            if args.save:
                state = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "loss": 1e10,
                    # "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                }
                filename = os.path.join('../experiments', args.experiment, 'checkpoints',
                                        "checkpoint_epoch_{:0>3d}_iter_{:}.pth.tar".format(epoch + 1,current_step))
                # save_checkpoint(state,is_best,filename)
                torch.save(state, filename)
        if i % 100 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                # f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                f"{i:5d}/{len(train_dataloader)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {rd_loss.item():.4f} | '
                f'MSE loss: {mse_loss.item():.4f} | '
                f'WARP loss: {warploss.item():.4f} | '
                f'Bpp loss: {bpp_loss.item():.4f} | '

            )
            logger_total.info(
                f"Train epoch {epoch}: ["
                # f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                f"{i:5d}/{len(train_dataloader)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {rd_loss.item():.4f} | '
                f'MSE loss: {mse_loss.item():.4f} | '
                f'WARP loss: {warploss.item():.4f} | '
                f'Bpp loss: {bpp_loss.item():.4f} | '

            )
            # if out_criterion["ms_ssim_loss"] is None:
            #     logger_train.info(
            #         f"Train epoch {epoch}: ["
            #         f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
            #         f" ({100. * i / len(train_dataloader):.0f}%)] "
            #         f'Loss: {rd_loss.item():.4f} | '
            #         f'MSE loss: {mse_loss.item():.4f} | '
            #         f'WARP loss: {warploss.item():.4f} | '
            #         f'Bpp loss: {bpp_loss.item():.2f} | '
            #
            #     )
            #     logger_total.info(
            #         f"Train epoch {epoch}: ["
            #         f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
            #         f" ({100. * i / len(train_dataloader):.0f}%)] "
            #         f'Loss: {rd_loss.item():.4f} | '
            #         f'MSE loss: {mse_loss.item():.4f} | '
            #         f'WARP loss: {warploss.item():.4f} | '
            #         f'Bpp loss: {bpp_loss.item():.2f} | '
            #
            #     )
            # else:
            #     logger_train.info(
            #         f"Train epoch {epoch}: ["
            #         f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
            #         f" ({100. * i / len(train_dataloader):.0f}%)] "
            #         f'Loss: {rd_loss.item():.4f} | '
            #         f'MS-SSIM loss: {ms_ssim_loss.item():.4f} | '
            #         f'Bpp loss: {bpp_loss.item():.2f} | '
            #
            #     )
            #     logger_total.info(
            #         f"Train epoch {epoch}: ["
            #         f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
            #         f" ({100. * i / len(train_dataloader):.0f}%)] "
            #         f'Loss: {rd_loss.item():.4f} | '
            #         f'MS-SSIM loss: {ms_ssim_loss.item():.4f} | '
            #         f'Bpp loss: {bpp_loss.item():.2f} | '
            #     )
    return current_step

def for_test_epoch(epoch, test_dataloader, model, save_dir, logger_val,logger_total, tb_logger, args):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()

    bpp_loss = AverageMeter()
    bpp_motion = AverageMeter()
    bpp_residual = AverageMeter()
    mse_loss = AverageMeter()

    ms_ssim_loss = AverageMeter()
    psnr = AverageMeter()
    wpsnr = AverageMeter()
    ms_ssim = AverageMeter()
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            d = [frames.to(device) for frames in batch]
            out_net = model(d)

            # out_criterion = criterion(out_net, d[1:])

            # rd_loss = args.lmbda * out_net['mse_loss'] + out_net["bpp"]
            bpp_loss.update(out_net["bpp"])
            bpp_motion.update(out_net["bpp_mv_y"]+out_net["bpp_mv_z"])
            bpp_residual.update(out_net["bpp_y"]+out_net["bpp_z"])
            loss.update(args.lmbda * out_net['mse_loss'] + out_net["bpp"])
            mse_loss.update(out_net['mse_loss'])

            # if out_criterion["mse_loss"] is not None:
            #     mse_loss.update(out_criterion["mse_loss"])
            # if out_criterion["ms_ssim_loss"] is not None:
            #     ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
            for i in range(d.__len__()-1):
                for j in range(d[i].shape[0]):
                    rec = torch2img(out_net['x_hat'][i][j])
                    warp = torch2img(out_net['x_warp'][i][j])
                    img = torch2img(d[i+1][j])
                    p, m = compute_metrics(rec, img)
                    w_p,_ = compute_metrics(warp, img)
                    psnr.update(p)
                    wpsnr.update(w_p)
                    ms_ssim.update(m)
                    if idx % 2 == 1:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        rec.save(os.path.join(save_dir, 'b%03d' % j+'_f%03d_rec.png' % i))
                        img.save(os.path.join(save_dir, 'b%03d' % j+'_f%03d_gt.png' % i))
                        warp.save(os.path.join(save_dir, 'b%03d' % j+'_f%03d_warp.png' % i))



    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_motion'), bpp_motion.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_residual'), bpp_residual.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: warp_psnr'), wpsnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)

    logger_val.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"MSE loss: {mse_loss.avg:.4f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"PSNR: {psnr.avg:.6f} | "
        f"WPSNR: {wpsnr.avg:.6f} | "
        f"MS-SSIM: {ms_ssim.avg:.6f}"
    )
    logger_total.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"MSE loss: {mse_loss.avg:.4f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"PSNR: {psnr.avg:.6f} | "
        f"WPSNR: {wpsnr.avg:.6f} | "
        f"MS-SSIM: {ms_ssim.avg:.6f}"
    )
    tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)
    # print(
    #     f"Test epoch {epoch}: Average losses:"
    #     f"\tLoss: {loss.avg:.3f} |"
    #     f"\tMSE loss: {mse_loss.avg:.3f} |"
    #     f"\tBpp loss: {bpp_loss.avg:.2f} |"
    # )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     default="dvc",
    #     choices=video_models.keys(),
    #     help="Model architecture (default: %(default)s)",
    # )
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=80,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-mf",
        "--multi-frames",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("-f","--finetune", action="store_true", help="Use finetune")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("-c","--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def seed_torch(seed=4096):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        seed_torch(args.seed)

    # Warning, the order of the transform composition should be kept.
    train_transforms = transforms.Compose(
        [transforms.ToTensor(),transforms.RandomCrop(args.patch_size)]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor(),transforms.CenterCrop(args.patch_size)]
    )

    # train_dataset = SCVFolder(
    #     args.dataset,
    #     rnd_interval=True,
    #     rnd_temp_order=True,
    #     split="train",
    #     transform=train_transforms,
    # )
    # test_dataset = SCVFolder(
    #     args.dataset,
    #     rnd_interval=False,
    #     rnd_temp_order=False,
    #     split="valid",
    #     transform=test_transforms,
    # )

    train_dataset = VideoFolder(
        args.dataset,
        args.multi_frames,
        rnd_interval=True,
        rnd_temp_order=True,
        split="sep_trainlist",
        transform=train_transforms,
    )
    test_dataset = VideoFolder(
        args.dataset,
        args.multi_frames,
        rnd_interval=False,
        rnd_temp_order=False,
        split="sep_testlist",
        transform=test_transforms,
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # net = video_models[args.model](quality=3)
    net = DMC()
    net = net.to(device)

    optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    if args.multi_frames > 2:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 55], gamma=0.1)
    # create log file
    if not os.path.exists(os.path.join('../experiments', args.experiment)):
        os.makedirs(os.path.join('../experiments', args.experiment))
    util.setup_logger('train', os.path.join('../experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    util.setup_logger('val', os.path.join('../experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    util.setup_logger('total', os.path.join('../experiments', args.experiment), 'total_' + args.experiment, level=logging.INFO,
                        screen=False, tofile=True)
    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    logger_total = logging.getLogger('total')
    method = os.path.split(os.path.split(os.getcwd())[-2])[-1]
    tb_logger = SummaryWriter(log_dir='../tb_logger/' +args.experiment +'_'+ method)
    logger_train.info(args)
    logger_train.info("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
    # logger_train.info(net)
    logger_total.info(args)
    logger_total.info("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
    os.makedirs(os.path.join('../experiments', args.experiment, 'checkpoints'), exist_ok=True)
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "epoch" in checkpoint:
            last_epoch = checkpoint["epoch"] + 1
            start_epoch = checkpoint["epoch"]
        else:
            last_epoch = 1
            start_epoch = 0
        if "state_dict" in checkpoint:
            net.load_state_dict(checkpoint["state_dict"])
        else:
            net.load_state_dict(checkpoint)
        #optimizer.load_state_dict(checkpoint["optimizer"])
        #lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
        best_loss = 1e10 #checkpoint['loss'] if checkpoint['loss'] is not None else 1e10
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    # best_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logger_total.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch(
            net,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,logger_total,
            tb_logger,
            current_step,
            args
        )
        save_dir = os.path.join('../experiments', args.experiment, 'val_images', '%03d' % (epoch + 1))
        loss = for_test_epoch(epoch, test_dataloader, net, save_dir, logger_val,logger_total, tb_logger,args)
        # lr_scheduler.step(loss)
        lr_scheduler.step()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        # if args.save:
        #     save_checkpoint(
        #         {
        #             "epoch": epoch,
        #             "state_dict": net.state_dict(),
        #             "loss": loss,
        #             "optimizer": optimizer.state_dict(),
        #             "lr_scheduler": lr_scheduler.state_dict(),
        #         },
        #         is_best,
        #     )
        #     if is_best:
        #         logger_val.info('best checkpoint saved.')
        if args.save:
            state = {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    # "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                }
            filename = os.path.join('../experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
            # save_checkpoint(state,is_best,filename)
            if (epoch + 1) % 2 ==0:
                torch.save(state, filename)

            if is_best:
                dest_filename = filename.replace(filename.split('/')[-1], "checkpoint_best_loss.pth.tar")
                torch.save(state, dest_filename)
                logger_val.info('best checkpoint  saved: Epoch{}'.format(epoch))
                logger_total.info('best checkpoint  saved: Epoch{}'.format(epoch))

if __name__ == "__main__":
    main(sys.argv[1:])
