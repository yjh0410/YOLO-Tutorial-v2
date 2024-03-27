# General Object Detection for Open World

## Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n odlab python=3.10
```

- Then, activate the environment:
```Shell
conda activate odlab
```

- Requirements:
1. Install necessary libraies
```Shell
pip install -r requirements.txt 
```

2. (optional) Compile MSDeformableAttention ops for DETR series

```bash
cd ./ppdet/modeling/transformers/ext_op/

python setup_ms_deformable_attn_op.py install
```
See [details](./models/detectors/rtdetr/basic_modules/ext_op/)

My environment:
- PyTorch = 2.2.0
- Torchvision = 0.17.0

At least, please make sure your torch is version 1.x.

## Experiments
### VOC
- Download VOC.
```Shell
cd <ODLab-World>
cd dataset/scripts/
sh VOC2007.sh
sh VOC2012.sh
```

- Check VOC
```Shell
cd <ODLab-World>
python dataset/voc.py
```

### COCO

- Download COCO.
```Shell
cd <ODLab-World>
cd dataset/scripts/
sh COCO2017.sh
```

- Clean COCO
```Shell
cd <ODLab-World>
cd tools/
python clean_coco.py --root path/to/coco --image_set val
python clean_coco.py --root path/to/coco --image_set train
```

- Check COCO
```Shell
cd <ODLab-World>
python dataset/coco.py
```

## Train 
We kindly provide a script `train.sh` to run the training code. You need to follow the following format to use this script：
```Shell
bash train.sh <model> <data> <data_path> <batch_size> <num_gpus> <master_port> <resume_weight>
```

For example, we use this script to train YOLO-N from the epoch-0:
```Shell
bash train.sh yolo_n coco path/to/coco 128 4 1699 None
```

We can also continue training from existing weights by passing the model's weight file to the resume parameter.
```Shell
bash train.sh yolo_n coco path/to/coco 128 4 1699 path/to/yolo_n.pth
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
```Shell
cd <ODLab-World>
cd config/data_config
```
You need to edit the `dataset_cfg` defined in `dataset_config.py`. You can refer to the `customed` defined in `dataset_cfg` to modify the relevant parameters, such as `num_classes`, `classes_names`, to adapt to our dataset.

For example:
```Shell
dataset_cfg = {
    'customed':{
        'data_name': 'AnimalDataset',
        'num_classes': 9,
        'class_indexs': (0, 1, 2, 3, 4, 5, 6, 7, 8),
        'class_names': ('bird', 'butterfly', 'cat', 'cow', 'dog', 'lion', 'person', 'pig', 'tiger', ),
    },
}
```

- Step-3: Convert customed to COCO format.

```Shell
cd <ODLab-World>
cd tools
# convert train split
python convert_ours_to_coco.py --root path/to/dataset/ --split train
# convert val split
python convert_ours_to_coco.py --root path/to/dataset/ --split val
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
cd <ODLab-World>
cd dataset
# convert train split
python customed.py --root path/to/dataset/ --split train
# convert val split
python customed.py --root path/to/dataset/ --split val
```

- Step-5 **Train**

For example:

```Shell
cd <ODLab-World>
python train.py --root path/to/dataset/ -d customed -m yolo_n -bs 16 -p path/to/yolo_n_coco.pth
```

- Step-6 **Test**

For example:

```Shell
cd <ODLab-World>
python test.py --root path/to/dataset/ -d customed -m yolo_n --weight path/to/checkpoint --show
```

- Step-7 **Eval**

For example:

```Shell
cd <ODLab-World>
python eval.py --root path/to/dataset/ -d customed -m yolo_n --weight path/to/checkpoint
```

## Deployment
1. [ONNX export and an ONNXRuntime](./deployment/ONNXRuntime/)
