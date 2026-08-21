#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HED-style contour extraction.

This script provides a self-contained HED-inspired implementation that follows
the core HED idea of multi-scale side outputs and fused edge prediction. If
trained HED Caffe weights are available, it can also run the original OpenCV DNN
pipeline through --proto and --model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


HED_MEAN_BGR = (104.00698793, 116.66876762, 122.67891434)


class CropLayer(object):
    """OpenCV DNN crop layer used by the original HED Caffe model."""

    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        input_shape, target_shape = inputs[0], inputs[1]
        batch_size, num_channels = input_shape[0], input_shape[1]
        height, width = target_shape[2], target_shape[3]
        self.ystart = int((input_shape[2] - target_shape[2]) / 2)
        self.xstart = int((input_shape[3] - target_shape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width
        return [[batch_size, num_channels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart : self.yend, self.xstart : self.xend]]


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-6:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = (image - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    height, width = magnitude.shape
    output = np.zeros((height, width), dtype=np.float32)
    degree = angle * 180.0 / np.pi
    degree[degree < 0] += 180

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            q = 0.0
            r = 0.0
            direction = degree[y, x]

            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                q = magnitude[y, x + 1]
                r = magnitude[y, x - 1]
            elif 22.5 <= direction < 67.5:
                q = magnitude[y + 1, x - 1]
                r = magnitude[y - 1, x + 1]
            elif 67.5 <= direction < 112.5:
                q = magnitude[y + 1, x]
                r = magnitude[y - 1, x]
            elif 112.5 <= direction < 157.5:
                q = magnitude[y - 1, x - 1]
                r = magnitude[y + 1, x + 1]

            if magnitude[y, x] >= q and magnitude[y, x] >= r:
                output[y, x] = magnitude[y, x]
    return output


def hed_style_edges(
    image_bgr: np.ndarray,
    scales: Iterable[float] = (1.0, 0.75, 0.5, 0.35),
    side_weights: Iterable[float] = (0.42, 0.28, 0.20, 0.10),
    high_threshold: int = 72,
    low_threshold: int = 28,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a soft edge map and a binary contour map."""
    original_h, original_w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge([lightness, channel_a, channel_b]), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    fused = np.zeros((original_h, original_w), dtype=np.float32)
    scale_list = list(scales)
    weight_list = list(side_weights)
    if len(scale_list) != len(weight_list):
        raise ValueError("scales and side_weights must have the same length")

    for index, (scale, weight) in enumerate(zip(scale_list, weight_list)):
        scaled_w = max(16, int(round(original_w * scale)))
        scaled_h = max(16, int(round(original_h * scale)))
        scaled_gray = cv2.resize(gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        blur_size = 3 + 2 * index
        blurred = cv2.GaussianBlur(scaled_gray, (blur_size, blur_size), 0)

        grad_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
        magnitude = cv2.magnitude(grad_x, grad_y)
        angle = cv2.phase(grad_x, grad_y, angleInDegrees=False)
        thinned = non_maximum_suppression(magnitude, angle)
        gradient_edge = normalize_to_uint8(thinned)

        canny_edge = auto_canny(blurred)
        side_output = cv2.addWeighted(gradient_edge, 0.75, canny_edge, 0.25, 0)
        side_output = cv2.resize(side_output, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        fused += weight * (side_output.astype(np.float32) / 255.0)

    fused = cv2.GaussianBlur(fused, (3, 3), 0)
    fused_uint8 = normalize_to_uint8(fused)

    _, strong = cv2.threshold(fused_uint8, high_threshold, 255, cv2.THRESH_BINARY)
    weak = cv2.inRange(fused_uint8, low_threshold, high_threshold - 1)
    connected = cv2.dilate(strong, np.ones((3, 3), np.uint8), iterations=1)
    binary = np.where((weak > 0) & (connected > 0), 255, strong).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return fused_uint8, binary


def opencv_dnn_hed(image_bgr: np.ndarray, proto: Path, model: Path) -> np.ndarray:
    """Run the original HED Caffe model through OpenCV DNN."""
    cv2.dnn_registerLayer("Crop", CropLayer)
    if hasattr(cv2.dnn, "readNetFromCaffe"):
        net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
    else:
        net = cv2.dnn.readNet(str(model), str(proto), "caffe")
    height, width = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image_bgr,
        scalefactor=1.0,
        size=(width, height),
        mean=HED_MEAN_BGR,
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    output = net.forward()
    edge = output[0, 0]
    edge = cv2.resize(edge, (width, height))
    return normalize_to_uint8(edge)


def resolve_input_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    # Handles common typo: D:\HED\1\_original.jpg while the file is D:\HED\1_original.jpg.
    parent = path.parent.parent if path.parent.name == "1" else path.parent
    fallback = parent / "1_original.jpg"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Input image not found: {path_text}")


def save_results(image_bgr: np.ndarray, soft: np.ndarray, binary: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = image_bgr.copy()
    overlay[binary > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(image_bgr, 0.72, overlay, 0.28, 0)

    cv2.imwrite(str(output_dir / "hed_soft_edges.png"), soft)
    cv2.imwrite(str(output_dir / "hed_binary_contours.png"), binary)
    cv2.imwrite(str(output_dir / "hed_overlay.png"), overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HED-style contour extraction")
    parser.add_argument("--input", default=r"D:\HED\1\_original.jpg", help="input image path")
    parser.add_argument("--output-dir", default=r"D:\HED\output", help="directory for results")
    parser.add_argument("--proto", default="", help="optional HED deploy.prototxt")
    parser.add_argument("--model", default="", help="optional HED caffe model")
    parser.add_argument("--high-threshold", type=int, default=72, help="binary contour high threshold")
    parser.add_argument("--low-threshold", type=int, default=28, help="binary contour low threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_dir = Path(args.output_dir)
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {input_path}")

    if args.proto and args.model:
        soft = opencv_dnn_hed(image_bgr, Path(args.proto), Path(args.model))
        _, binary = cv2.threshold(soft, args.high_threshold, 255, cv2.THRESH_BINARY)
    else:
        soft, binary = hed_style_edges(
            image_bgr,
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
        )

    save_results(image_bgr, soft, binary, output_dir)
    print(f"input: {input_path}")
    print(f"soft edge map: {output_dir / 'hed_soft_edges.png'}")
    print(f"binary contours: {output_dir / 'hed_binary_contours.png'}")
    print(f"overlay preview: {output_dir / 'hed_overlay.png'}")


if __name__ == "__main__":
    main()
