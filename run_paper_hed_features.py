#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the original HED Caffe model and export side-output feature maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hed_edge import CropLayer, HED_MEAN_BGR, normalize_to_uint8, resolve_input_path


FEATURE_BLOBS = [
    "sigmoid-dsn1",
    "sigmoid-dsn2",
    "sigmoid-dsn3",
    "sigmoid-dsn4",
    "sigmoid-dsn5",
    "sigmoid-fuse",
]


def read_hed_net(proto: Path, model: Path) -> cv2.dnn.Net:
    cv2.dnn_registerLayer("Crop", CropLayer)
    return cv2.dnn.readNetFromCaffe(str(proto), str(model))


def save_feature_map(feature: np.ndarray, image_size: tuple[int, int], path: Path) -> np.ndarray:
    width, height = image_size
    feature_2d = np.squeeze(feature)
    if feature_2d.ndim != 2:
        raise ValueError(f"Expected 2D feature map after squeeze, got {feature_2d.shape}")
    feature_2d = cv2.resize(feature_2d, (width, height), interpolation=cv2.INTER_LINEAR)
    feature_u8 = normalize_to_uint8(feature_2d)
    cv2.imwrite(str(path), feature_u8)
    return feature_u8


def colorize_edge_map(edge_u8: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(edge_u8, cv2.COLORMAP_TURBO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export original HED side-output feature maps")
    parser.add_argument("--input", default=r"D:\HED\1\_original.jpg", help="input image")
    parser.add_argument(
        "--proto",
        default=r"D:\HED\hed-master-unpacked\hed-master\examples\hed\deploy.prototxt",
        help="HED deploy.prototxt",
    )
    parser.add_argument(
        "--model",
        default=r"D:\HED\hed-master-unpacked\hed-master\examples\hed\hed_pretrained_bsds.caffemodel",
        help="HED pretrained caffemodel",
    )
    parser.add_argument("--output-dir", default=r"D:\HED\paper_hed_features", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = resolve_input_path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]

    net = read_hed_net(Path(args.proto), Path(args.model))
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(width, height),
        mean=HED_MEAN_BGR,
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    outputs = net.forward(FEATURE_BLOBS)

    summary = []
    for blob_name, feature in zip(FEATURE_BLOBS, outputs):
        feature_path = output_dir / f"{blob_name}.png"
        heatmap_path = output_dir / f"{blob_name}_heatmap.png"
        feature_u8 = save_feature_map(feature, (width, height), feature_path)
        cv2.imwrite(str(heatmap_path), colorize_edge_map(feature_u8))
        summary.append((blob_name, feature.shape, int(feature_u8.min()), int(feature_u8.max()), float(feature_u8.mean())))

    fuse = cv2.imread(str(output_dir / "sigmoid-fuse.png"), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(fuse, 72, 255, cv2.THRESH_BINARY)
    overlay = image.copy()
    overlay[binary > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(image, 0.72, overlay, 0.28, 0)
    cv2.imwrite(str(output_dir / "sigmoid-fuse_binary.png"), binary)
    cv2.imwrite(str(output_dir / "sigmoid-fuse_overlay.png"), overlay)

    print(f"input: {image_path}")
    print(f"output_dir: {output_dir}")
    for blob_name, shape, min_value, max_value, mean_value in summary:
        print(f"{blob_name}: raw_shape={shape}, uint8_min={min_value}, uint8_max={max_value}, uint8_mean={mean_value:.2f}")


if __name__ == "__main__":
    main()
