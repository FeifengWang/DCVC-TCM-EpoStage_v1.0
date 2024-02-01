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

"""
Update the CDFs parameters of a trained model.

To be called on a model checkpoint after training. This will update the internal
CDFs related buffers required for entropy coding.
"""
import argparse
import hashlib
import os.path
import sys

from pathlib import Path
from typing import Dict

import torch

from compressai.models.google import (
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    MeanScaleHyperprior,
    ScaleHyperprior,
)
from compressai.models.video.google import ScaleSpaceFlow
from compressai.zoo import load_state_dict
from compressai.zoo.image import model_architectures as zoo_models

from model.model import DSCVC
def sha256_file(filepath: Path, len_hash_prefix: int = 8) -> str:
    # from pytorch github repo
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        while True:
            buf = f.read(8192)
            if len(buf) == 0:
                break
            sha256.update(buf)
    digest = sha256.hexdigest()

    return digest[:len_hash_prefix]


def load_checkpoint(filepath: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(filepath, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = load_state_dict(state_dict)
    return state_dict


description = """
Export a trained model to a new checkpoint with an updated CDFs parameters and a
hash prefix, so that it can be loaded later via `load_state_dict_from_url`.
""".strip()

models = {
    "factorized-prior": FactorizedPrior,
    "jarhp": JointAutoregressiveHierarchicalPriors,
    "mean-scale-hyperprior": MeanScaleHyperprior,
    "scale-hyperprior": ScaleHyperprior,
    "ssf2020": ScaleSpaceFlow,

}
models.update(zoo_models)


def setup_args():
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument(
    #     "filepath", type=str,default='/bao/wff/DVC/CompressAI-base/experiments/exp_01_mse_q3/checkpoints/checkpoint_best_loss.pth.tar', help="Path to the checkpoint model to be exported."
    # )
    parser.add_argument("-n", "--name", type=str, help="Exported model name.")
    parser.add_argument("-d", "--dir", type=str, help="Exported model directory.")
    parser.add_argument(
        "--no-update",
        action="store_true",
        default=False,
        help="Do not update the model CDFs parameters.",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        default="scale-hyperprior",
        choices=models.keys(),
        help="Set model architecture (default: %(default)s).",
    )
    parser.add_argument(
        "-exp", "--experiment", type=str, help="Experiment name"
    )
    parser.add_argument(
        "--epoch", type=int, default=-1, help="Epoch"
    )
    parser.add_argument(
        "--ckpt_name", type=str, help="Experiment name"
    )
    return parser


def main(argv):
    args = setup_args().parse_args(argv)
    if args.epoch != -1:
        filepath = os.path.join('../experiments',args.experiment,'checkpoints','checkpoint_%03d.pth.tar'%args.epoch)
    else:
        filepath = os.path.join('../experiments', args.experiment, 'checkpoints',
                                'checkpoint_best_loss.pth.tar')
    if args.ckpt_name is not None:
        filepath = os.path.join('../experiments/', args.experiment, 'checkpoints',args.ckpt_name)
    filepath = Path(filepath).resolve()
    if not filepath.is_file():
        raise RuntimeError(f'"{filepath}" is not a valid file.')

    state_dict = load_checkpoint(filepath)

    # model_cls_or_entrypoint = models[args.architecture]
    # if not isinstance(model_cls_or_entrypoint, type):
    #     model_cls = model_cls_or_entrypoint()
    # else:
    #     model_cls = model_cls_or_entrypoint
    model_cls_or_entrypoint = DSCVC()
    model_cls = model_cls_or_entrypoint
    net = model_cls.from_state_dict(state_dict)

    if not args.no_update:
        net.update(force=True)
    state_dict = net.state_dict()

    if not args.name:
        filename = filepath
        while filename.suffixes:
            filename = Path(filename.stem)
    else:
        filename = args.name

    ext = "".join(filepath.suffixes)

    # if args.dir is not None:
    #     output_dir = Path(args.dir)
    #     Path(output_dir).mkdir(exist_ok=True)
    # else:W
    #     output_dir = Path.cwd()
    output_dir = os.path.join('../experiments',args.experiment,'checkpoint_updated')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir,f))
    filepath = Path(f"{output_dir}/{filename}{ext}")
    torch.save(state_dict, filepath)
    hash_prefix = sha256_file(filepath)

    filepath.rename(f"{output_dir}/{filename}-{hash_prefix}{ext}")


if __name__ == "__main__":
    main(sys.argv[1:])
