#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Thanks to YOLOX: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/export_onnx.py

import argparse
import os
from loguru import logger
import sys
sys.path.append('..')

import torch
from torch import nn

from utils.misc import SiLU
from utils.misc import load_weight, replace_module

from config import build_config
from models import build_model


def make_parser():
    parser = argparse.ArgumentParser("FreeYOLO ONNXRuntime")
    # basic
    parser.add_argument('--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument("--input", default="images", type=str,
                        help="input node name of onnx model")
    parser.add_argument("--output", default="output", type=str,
                        help="output node name of onnx model")
    parser.add_argument("--opset", default=13, type=int,
                        help="onnx opset version")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--dynamic", action="store_true", default=False,
                        help="whether the input shape should be dynamic or not")
    parser.add_argument("--no-onnxsim", action="store_true", default=False,
                        help="use onnxsim or not")
    parser.add_argument("-f", "--exp_file", default=None, type=str,
                        help="experiment description file")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    # model
    parser.add_argument('--model', default='yolov8_n', type=str,
                        help='build FreeYOLOv2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))

    # Build config
    cfg = build_config(args)
    cfg.num_classes = 80  # for coco

    # Build model
    model = build_model(args, cfg, is_val=False)

    # Load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.eval()

    logger.info(" => loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)

    # save onnx file
    save_path = os.path.join(os.path.split(args.weight)[0], str(args.opset))
    os.makedirs(save_path, exist_ok=True)
    output_name = os.path.join(args.model + '.onnx')
    output_path = os.path.join(save_path, output_name)

    torch.onnx._export(
        model,
        dummy_input,
        output_path,
        input_names=[args.input],
        output_names=[output_name],
        dynamic_axes={args.input: {0: 'batch'},
                      output_name: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )

    logger.info("generated onnx model named {}".format(output_path))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(output_path)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"

        # save onnxsim file
        save_path = os.path.join(save_path, 'onnxsim')
        os.makedirs(save_path, exist_ok=True)
        output_path = os.path.join(save_path, output_name)
        onnx.save(model_simp, output_path)
        logger.info("generated simplified onnx model named {}".format(output_path))


if __name__ == "__main__":
    main()
