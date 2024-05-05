# 《YOLO目标检测》书籍的第二版源代码

本项目是《YOLO目标检测》书籍第二版的源代码，包含了此书所涉及到的所有YOLO模型、RT-DETR模型、DETR模型、FCOS模型，以及YOLOF模型。对于YOLO和RT-DETR，
读者可以在该项目的`yolo/`文件夹下找到所有的源代码；对于DETR、FCOS和YOLOF模型，读者可以在该项目的`odlab/`文件夹下找到所有的源代码。

源代码由两种模式构成：

- 第一种是“**章节**”模式，即在该项目的`chapters/`文件夹下，包含了书中每个章节所介绍到的源代码，均可进行调试，以便读者充分理解这些代码的原理和效果；但这种模式虽然可以帮助读者理解每一章节的内容，但无法帮助读者构建起对基于深度学习技术的目标检测框架的更加完整的、成体系的认知，因此，本项目也同时提供了第二种模式；

- 第二种是“**项目**”模式，即在该项目的`yolo/`和`odlab/`文件夹下，都以笔者所偏好的代码风格来构建了较为完整的目标检测项目代码，以便读者可以更深入地理解一个较为完整的基于深度学习技术的目标检测项目代码的结构，帮助读者初步地建立起面向“项目”层次的，同时，为读者未来更深入地接触其他的项目代码提供了一些实践基础，但这种模式可能对处在入门阶段读者造成一些上手代码的困难，因此，建议读者先遵循第一种的“章节”模式，配合图书内容并实操对应的代码来加深对基础知识的了解，随后，在学到书中的训练和测试的相关内容时，切换“项目”模式，查阅相关的训练代码和测试代码，更加深入的了解基于深度学习技术的目标检测框架。

## 准备工作
在使用此代码前，需要读者完成一些必要的环境配置，如python语言的安装、pytorch框架的安装等，随后，遵循`yolo/`和`odlab/`两个文件中的`README.md`文件所提供的内容，配置相关的环境、准备学习所需的数据集，并了解如何使用此项目代码进行训练和测试。如果读者想使用此代码去训练自定义的数据集，也请遵从这两个文件夹中的`README.md`文件中所给出的指示和说明来准备数据，并训练和测试。

## 实验结果
### YOLO系列
下面的两个表分别汇报了本项目的YOLO系列的small量级的模型在VOC和COCO数据集上的性能指标，

- VOC

| Model       | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|-------------|-------|-------|-------------------|--------|--------|
| YOLOv1-R18  | 1xb16 |  640  |               | [ckpt]() | [log]() |
| YOLOv2-R18  | 1xb16 |  640  |               | [ckpt]() | [log]() |
| YOLOv3-S    | 1xb16 |  640  |               | [ckpt]() | [log]() |
| YOLOv5-S    | 1xb16 |  640  |               | [ckpt]() | [log]() |
| YOLOv5-AF-S | 1xb16 |  640  |               | [ckpt]() | [log]() |
| YOLOv8-S    | 1xb16 |  640  |               | [ckpt]() | [log]() |
| GELAN-S     | 1xb16 |  640  |               | [ckpt]() | [log]() |

- COCO

| Model       | Batch | Scale | FPS<sup>FP32<br>RTX 4060 |AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight | Logs |
|-------------|-------|-------|--------------------------|------------------------|-------------------|-------------------|--------------------|--------|------|
| YOLOv1-R18  | 1xb16 |  640  |                          |         27.6           |       46.8        |   37.8            |   21.3             | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/yolov1_r18_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/YOLOv1-R18-COCO.txt) |
| YOLOv2-R18  | 1xb16 |  640  |                          |         28.4           |       47.4        |   38.0            |   21.5             | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/yolov2_r18_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/YOLOv2-R18-COCO.txt) |
| YOLOv3-S    | 1xb16 |  640  |                          |         31.3           |        49.2       |   25.2            |   7.3              | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v3/releases/download/yolo_tutorial_ckpt/yolov3_s_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v3/releases/download/yolo_tutorial_ckpt/YOLOv3-S-COCO.txt) |
| YOLOv5-S    | 1xb16 |  640  |                          |       38.8             |     56.9          |   27.3            |   9.0              | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/yolov5_s_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/YOLOv5-S-COCO.txt) |
| YOLOv5-AF-S | 1xb16 |  640  |                          |       39.6             |       58.7        |   26.9            |   8.9              | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/yolov5_af_s_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/YOLOv5-AF-S-COCO.txt) |
| YOLOv8-S    | 1xb16 |  640  |                          |                        |                   |   28.4            |   11.3            |  |  |
| GELAN-S     | 1xb16 |  640  |                          |                        |                   |   26.9            |   8.9             |  |  |

### RT-DETR系列
下表汇报了本项目的RT-DETR系列在COCO数据集上的性能指标，

- COCO

| Model        | Batch | Scale | FPS<sup>FP32<br>RTX 4060 |AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight | Logs |
|--------------|-------|-------|--------------------------|------------------------|-------------------|-------------------|--------------------|--------|------|
| RT-DETR-R18  | 4xb4  |  640  |                          |           45.5         |        63.5       |        66.8       |        21.0        | [ckpt](https://github.com/yjh0410/ODLab-World/releases/download/coco_weight/rtdetr_r18_coco.pth) | [log](https://github.com/yjh0410/ODLab-World/releases/download/coco_weight/RT-DETR-R18-COCO.txt)|
| RT-DETR-R50  | 4xb4  |  640  |                          |           50.6         |        69.4       |       112.1       |        36.7        | [ckpt](https://github.com/yjh0410/ODLab-World/releases/download/coco_weight/rtdetr_r50_coco.pth) | [log](https://github.com/yjh0410/ODLab-World/releases/download/coco_weight/RT-DETR-R50-COCO.txt)|

### ODLab系列
下表汇报了本项目的ODLab系列在COCO数据集上的性能指标，

- COCO

| Model          | Sclae      | FPS<sup>FP32<br>RTX 4060 | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs |
|----------------|------------|--------------------------|------------------------|-------------------|--------|------|
| FCOS_R18_1x    |  800,1333  |           24             |          34.0          |        52.2       | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/fcos_r18_1x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/FCOS-R18-1x.txt) |
| FCOS_R50_1x    |  800,1333  |            9             |          39.0          |        58.3       | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/fcos_r50_1x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/FCOS-R50-1x.txt) |
| FCOS_RT_R18_3x |  512,736   |           56             |          35.8          |        53.3       | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/fcos_rt_r18_3x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/FCOS-RT-R18-3x.txt) |
| FCOS_RT_R50_3x |  512,736   |           34             |          40.7          |        59.3       | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/fcos_rt_r50_3x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/FCOS-RT-R50-3x.txt) |
| YOLOF_R18_C5_1x  |  800,1333  |          54          |          32.8          |       51.4        | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/yolof_r18_c5_1x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/YOLOF-R18-C5-1x.txt) |
| YOLOF_R50_C5_1x  |  800,1333  |          21          |          37.7          |       57.2        | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/yolof_r50_c5_1x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/YOLOF-R50-C5-1x.txt) |


# The source code of the second edition of the book "YOLO Object Detection"
This project is the source code of the "YOLO Target Detection" book （second edition）, which includes all YOLO models, RT-DETR models, DETR models, FCOS models, and YOLOF models involved in this book. For YOLO and RT-DETR, readers can find all source codes in the `yolo/` folder of the project; for DETR, FCOS and YOLOF models, readers can find all source codes in the `odlab/` folder of the project. 

The source code consists of two modes:

- The first mode is the "**Chapter**" mode, where the `chapters/` folder of this project contains the source code introduced in each chapter of the book. Readers can debug the code to fully understand its principles and effects. However, although this mode helps readers understand the content of each chapter, it does not assist in building a more comprehensive and systematic understanding of object detection frameworks based on deep learning techniques. Therefore, this project also provides a second mode.

- The second mode is the "**Project**" mode, where the `yolo/` and `odlab/` folders of this project contain a more complete set of object detection project code, constructed in the coding style preferred by the author. This allows readers to gain a deeper understanding of the structure of a complete object detection project based on deep learning techniques. It helps readers to establish a preliminary understanding at the "project" level and provides a practical foundation for readers to delve into other project codes in the future. However, this mode may pose some challenges for readers who are at the beginner stage and are not familiar with project-level code. Therefore, it is recommended that readers initially follow the first mode, the "Chapter" mode, and work through the corresponding code examples while studying the book's content to deepen their understanding of the fundamental concepts. Later, when learning about training and testing in the book, readers can switch to the "Project" mode and refer to the relevant training and testing code to gain a more in-depth understanding of object detection frameworks based on deep learning techniques.

## Preparation
Before using this code, readers are required to complete some necessary environment configurations, such as installing the Python language and the PyTorch framework. Afterwards, following the instructions provided in the README.md files in the yolo/ and odlab/ directories, readers should configure the relevant environment, prepare the required datasets, and learn how to use this project code for training and testing. If readers want to train on custom datasets using this code, they should also follow the instructions and guidelines provided in the README.md files in these two directories to prepare the data and conduct training and testing.
