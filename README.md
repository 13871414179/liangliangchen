# HED Edge Detection Demo

本项目基于论文 **Holistically-Nested Edge Detection, ICCV 2015** 的官方 Caffe 网络结构与预训练权重，完成图像轮廓/边缘提取，并导出 HED 的 5 个 side-output 特征图和最终融合边缘图。

## 项目内容

- `hed_edge.py`：运行 HED 融合输出，生成灰度边缘图、二值轮廓图和叠加预览图。
- `run_paper_hed_features.py`：导出 HED 的 5 个 side-output 特征图和 `sigmoid-fuse` 融合图。
- `deploy.prototxt`：HED 官方 Caffe 网络结构文件。
- `docs/`：算法流程技术文档，包含 Markdown 和 Word 版本。
- `assets/hed_paper_model_structure.png`：HED 论文模型结构图。
- `examples/`：示例输入图像和本项目已生成的输出结果。

## 模型权重

预训练权重文件没有直接放入仓库。请从 HED 官方地址下载：

```text
https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel
```

下载后建议放在：

```text
models/hed_pretrained_bsds.caffemodel
```

也可以在运行命令中通过 `--model` 指定其他位置。

## 环境安装

建议使用 Python 3.9，并安装依赖：

```bash
pip install -r requirements.txt
```

注意：OpenCV 5 已移除 Caffe importer，无法读取 `.caffemodel`。本项目建议使用 OpenCV 4.x。

## 运行最终边缘提取

```bash
python hed_edge.py ^
  --input "examples/1_original.jpg" ^
  --output-dir "examples/paper_hed_output" ^
  --proto "deploy.prototxt" ^
  --model "models/hed_pretrained_bsds.caffemodel"
```

输出：

```text
hed_soft_edges.png
hed_binary_contours.png
hed_overlay.png
```

## 导出 HED 特征图

```bash
python run_paper_hed_features.py ^
  --input "examples/1_original.jpg" ^
  --output-dir "examples/paper_hed_features" ^
  --proto "deploy.prototxt" ^
  --model "models/hed_pretrained_bsds.caffemodel"
```

输出：

```text
sigmoid-dsn1.png
sigmoid-dsn2.png
sigmoid-dsn3.png
sigmoid-dsn4.png
sigmoid-dsn5.png
sigmoid-fuse.png
```

其中：

- `sigmoid-dsn1`：浅层细节边缘，细节多，也更容易包含纹理。
- `sigmoid-dsn2`：浅层稳定边缘。
- `sigmoid-dsn3`：中层结构边缘。
- `sigmoid-dsn4`：深层语义边界。
- `sigmoid-dsn5`：高层整体轮廓。
- `sigmoid-fuse`：5 个 side-output 融合后的最终边缘概率图。

## 算法一句话说明

HED 可以理解成一个“学过很多人工轮廓样例的自动描边工具”。它不是只找图片里颜色变化大的地方，而是先看细节，再看整体形状，最后把多层判断融合起来，输出更接近真实物体边界的轮廓图。

## 参考资料

- Saining Xie, Zhuowen Tu. **Holistically-Nested Edge Detection**. ICCV 2015.
- Paper: https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html
- Official code: https://github.com/s9xie/hed
