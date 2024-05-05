# YOLO系列教程
这部分是YOLO系列教程的项目代码，同时，RT-DETR系列也包含在本项目中。

## 配置环境
- 首先，创建一个conda虚拟环境，例如，我们创建一个名为`yolo_tutorial`的虚拟环境，并设置python版本为3.10：
```Shell
conda create -n yolo_tutorial python=3.10
```

- 随后，激活创建好的虚拟环境
```Shell
conda activate yolo_tutorial
```

- 然后，安装本项目所用到的各种python包和库
1. 安装必要的各种包和库
```Shell
pip install -r requirements.txt 
```

2. (可选) 为了能够加快RT-DETR模型中的可形变自注意力计算，可以考虑编译 `MSDeformableAttention` 的CUDA算子

```bash
cd models/rtdetr/basic_modules/ext_op/
python setup_ms_deformable_attn_op.py install
```

下面是笔者常用的环境配置，可供读者参考:
- PyTorch = 2.2.0
- Torchvision = 0.17.0

如果你的设备不支持2.x版本的torch，可以自行安装其他版本的，确保torch版本是在1.0以上即可。

## 实验内容
### 准备 VOC 数据
- 下载VOC2007 和VOC2012数据.
```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset/scripts/
sh VOC2007.sh
sh VOC2012.sh
```

- 检查VOC数据
```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset/
python voc.py --is_train --aug_type yolo
```

### COCO

- 下载 COCO 2017 数据.
```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset/scripts/
sh COCO2017.sh
```

- 检查 COCO 数据
```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset/
python coco.py --is_train --aug_type yolo
```

## Train
对于训练，我们提供了一个名为`train.sh`的脚本，方便读者可以一键启动命令，不过，为了顺利使用这个脚本，该脚本需要接受一些命令行参数，读者可以参考下面的格式来正确使用本训练脚本：

```Shell
bash train.sh <model> <data> <data_path> <batch_size> <num_gpus> <master_port> <resume_weight>
```
其中，<model>是要训练的模型名称；<data>是数据集名称；<data_path>是数据集的存放路径；<batch_size>顾名思义；<num_gpus>顾名思义；<master_port>DDP所需的port值，随意设定即可，如1234，4662；<resume_weight>已保存的checkpoint，训练中断后继续训练时会用到，如果从头训练，则设置为`None`即可。

例如，我们使用该脚本来训练本项目的`YOLOv1-R18`模型，使用4张GPU，从头开始训练：
```Shell
bash train.sh yolov1_r18 coco path/to/coco 128 4 1699 None
```

假如训练中断了，我们要接着继续训练，则参考下面的命令：
```Shell
bash train.sh yolov1_r18 coco path/to/coco 128 4 1699 path/to/yolov1_r18_coco.pth
```
其中，最后的命令行参数`path/to/yolov1_r18_coco.pth`是在上一次训练阶段中已保存的checkpoint文件。

## 训练自定义数据
除了本教程所介绍的VOC和COCO两大主流数据集，本项目也支持训练读者自定义的数据。不过，需要按照本项目的要求来从头开始准备数据，包括标注和格式转换（COCO格式）。如果读者手中的数据已经都准备好了，倘若不符合本项目的格式，还请另寻他法，切不可强行使用本项目，否则出了问题，我们也无法提供解决策略，只能后果自负。为了能够顺利使用本项目，请读者遵循以下的步骤来开始准备数据

- 第1步，准备图片，诸如jpg格式、png格式等都可以，构建为自定义数据集，不妨起名为`CustomedDataset`，然后使用开源的`labelimg`制作标签文件。有关于`labelimg`的使用方法，请自行了解。完成标注后，应得到如下所示的文件目录格式：

```
CustomedDataset
|_ train
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|_ val
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|  ...
```

- 第2步: 修改与数据有关的配置参数
读者需要修改定义在`dataset/customed.py`文件中的`customed_class_indexs` 和 `customed_class_labels`两个参数，前者是类别索引，后者是类别名称。例如，我们使用了如下的定义以供读者参考:
```Shell
# dataset/customed.py
customed_class_indexs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
customed_class_labels = ('bird', 'butterfly', 'cat', 'cow', 'dog', 'lion', 'person', 'pig', 'tiger', )
```

- 第3步: 将数据转换为COCO的json格式
尽管已经使用`labelimg`软件标准了XML格式的标签文件，不过，为了能够顺利使用coco的evaluation工具，我们建议进一步将数据的格式转换为coco的json格式，具体操作如下所示，分别准备好`train`的json文件和`val`的json文件。

```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd tools
# convert train split
python convert_ours_to_coco.py --root path/to/customed_dataset/ --split train
# convert val split
python convert_ours_to_coco.py --root path/to/customed_dataset/ --split val
```

随后，我们便可得到一个名为`train.json` 的文件和一个名为 `val.json` 文件，如下所示.
```
CustomedDataset
|_ train
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|     |_ train.json
|_ val
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|     |_ val.json
|  ...
```

- 第4步：检查数据
然后，我们来检查数据是否准备完毕，对此，可以通过运行读取数据的代码文件来查看数据可视化结果，如果能顺利看到数据可视化结果，表明数据已准备完毕。读者可参考如下所示的运行命令。

```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset
# convert train split
python customed.py --root path/to/customed_dataset/ --split train
# convert val split
python customed.py --root path/to/customed_dataset/ --split val
```

- 第5步：使用自定义数据训练模型
接下来，我们就可以使用自定义的数据训练本项目的模型，例如，训练`YOLOv1-R18`模型，可参考如下的运行命令：

```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
bash train.sh yolov1_r18 customed path/to/customed_dataset/ 128 4 1699 None
```

- 第6步：使用自定义数据测试模型

我们就可以使用自定义的数据测试已训练好的模型，例如，测试`YOLOv1-R18`模型，观察检测结果的可视化图像，可参考如下的运行命令：

```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
python test.py -d customed --root path/to/customed_dataset/ -d customed -m yolov1_r18 --weight path/to/checkpoint --show
```

- 第7步：使用自定义数据验证模型

我们就可以使用自定义的数据验证已训练好的模型，例如，验证`YOLOv1-R18`模型的性能，得到AP指标，可参考如下的运行命令：

```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
python eval.py -d customed --root path/to/customed_dataset/ -d customed -m yolov1_r18 --weight path/to/checkpoint
```

## Demo
本项目提供了用于检查本地图片的代码文件`demo.py`，支持使用VOC/COCO/自定义数据所训练出来的模型来检查本地的图片/视频/笔记本摄像头。

例如，检查本地的图片，使用COCO数据训练的模型`YOLOv1_R18`：
```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               --cuda \
               --img_size 640 \
               --model yolov1_r18 \
               --weight path/to/weight \
               --dataset coco \
               --show
```

例如，检查本地的视频，使用COCO数据训练的模型`YOLOv1_R18`，并将视频结果保存为GIF图片：
```Shell
python demo.py --mode video \
               --path_to_vid data/demo/video \
               --cuda \
               --img_size 640 \
               --model yolov1_r18 \
               --weight path/to/weight \
               --dataset coco \
               --show \
               --gif
```

例如，检查笔记本的摄像头做实时检测，使用COCO数据训练的模型`YOLOv1_R18`，并将视频结果保存为GIF图片：
```Shell
python demo.py --mode camera \
               --cuda \
               --img_size 640 \
               --model yolov1_r18 \
               --weight path/to/weight \
               --dataset coco \
               --show \
               --gif
```
如果是外接的摄像头，需要读者略微调整`demo.py`文件中的代码，如下所示：
```Shell
    # --------------- Camera ---------------
    if mode == 'camera':
        ...

        # 笔记本摄像头，index=0；外接摄像头，index=1；
        cap = cv2.VideoCapture(index=0, apiPreference=cv2.CAP_DSHOW)
    
    ...
```


-------------------

# Tutorial of YOLO series

## Requirements
- We recommend you to use Anaconda to create a conda environment. For example, we create a one named `yolo_tutorial` with python=3.10:
```Shell
conda create -n yolo_tutorial python=3.10
```

- Then, activate the environment:
```Shell
conda activate yolo_tutorial
```

- Requirements:
1. Install necessary libraies
```Shell
pip install -r requirements.txt 
```

2. (optional) Compile `MSDeformableAttention` ops for RT-DETR series

```bash
cd models/rtdetr/basic_modules/ext_op/
python setup_ms_deformable_attn_op.py install
```

My environment:
- PyTorch = 2.2.0
- Torchvision = 0.17.0

At least, please make sure your torch >= 1.0.

## Experiments
### VOC
- Download VOC.
```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset/scripts/
sh VOC2007.sh
sh VOC2012.sh
```

- Check VOC
```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset/
python voc.py --is_train --aug_type yolo
```

### COCO

- Download COCO.
```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset/scripts/
sh COCO2017.sh
```

- Check COCO
```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset/
python coco.py --is_train --aug_type yolo
```

## Train 
We kindly provide a script `train.sh` to run the training code. You need to follow the following format to use this script：
```Shell
bash train.sh <model> <data> <data_path> <batch_size> <num_gpus> <master_port> <resume_weight>
```

For example, we use this script to train `YOLOv1-R18` from the epoch-0 with 4 gpus:
```Shell
bash train.sh yolov1_r18 coco path/to/coco 128 4 1699 None
```

We can also continue training from the existing weight by passing the model's weight file to the resume parameter.
```Shell
bash train.sh yolov1_r18 coco path/to/coco 128 4 1699 path/to/yolov1_r18_coco.pth
```


## Train on custom dataset
Besides the popular datasets, we can also train the model on ourself dataset. To achieve this goal, you should follow these steps:
- Step-1: Prepare the images (JPG/JPEG/PNG ...) and use `labelimg` to make XML format annotation files.

```
CustomedDataset
|_ train
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|_ val
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|  ...
```

- Step-2: Make the configuration for our dataset.
You need to edit the `customed_class_indexs` and `customed_class_labels` defined in `dataset/customed.py` to adapt to your customed dataset.

For example:
```Shell
# dataset/customed.py
customed_class_indexs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
customed_class_labels = ('bird', 'butterfly', 'cat', 'cow', 'dog', 'lion', 'person', 'pig', 'tiger', )
```

- Step-3: Convert customed to COCO format.

```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd tools
# convert train split
python convert_ours_to_coco.py --root path/to/customed_dataset/ --split train
# convert val split
python convert_ours_to_coco.py --root path/to/customed_dataset/ --split val
```
Then, we can get a `train.json` file and a `val.json` file, as shown below.
```
CustomedDataset
|_ train
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|     |_ train.json
|_ val
|  |_ images     
|     |_ 0.jpg
|     |_ 1.jpg
|     |_ ...
|  |_ annotations
|     |_ 0.xml
|     |_ 1.xml
|     |_ ...
|     |_ val.json
|  ...
```

- Step-4 Check the data.

```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
cd dataset
# convert train split
python customed.py --root path/to/customed_dataset/ --split train
# convert val split
python customed.py --root path/to/customed_dataset/ --split val
```

- Step-5 **Train** on the customed dataset

For example:

```Shell
# With coco pretrained weight
cd <YOLO-TUTORIAL-V2/yolo/>
python train.py --root path/to/customed_dataset/ -d customed -m yolov1_r18 -bs 16 -p path/to/yolov1_r18_coco.pth
```

```Shell
# Without coco pretrained weight
cd <YOLO-TUTORIAL-V2/yolo/>
python train.py --root path/to/customed_dataset/ -d customed -m yolov1_r18 -bs 16
```

- Step-6 **Test** on the customed dataset

For example:

```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
python test.py --root path/to/customed_dataset/ -d customed -m yolov1_r18 --weight path/to/checkpoint --show
```

- Step-7 **Eval** on the customed dataset

For example:

```Shell
cd <YOLO-TUTORIAL-V2/yolo/>
python eval.py --root path/to/customed_dataset/ -d customed -m yolov1_r18 --weight path/to/checkpoint
```

## Demo
I have provide some images in `data/demo/images/`, so you can run following command to run a demo with coco pretrained model:

```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               --cuda \
               --img_size 640 \
               --model yolov1_r18 \
               --weight path/to/weight \
               --dataset coco \
               --num_classes 80 \
               --show
```

If you want to try this command with voc pretrained model, you could refer to the following command:
```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               --cuda \
               --img_size 640 \
               --model yolov1_r18 \
               --weight path/to/weight \
               --dataset voc \
               --num_classes 20 \
               --show
```


If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_vid data/demo/videos/your_video \
               --cuda \
               --img_size 640 \
               --model yolov1_r18 \
               --weight path/to/weight \
               --show \
               --gif
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               --cuda \
               --img_size 640 \
               --model yolov1_r18 \
               --weight path/to/weight \
               --show \
               --gif
```
