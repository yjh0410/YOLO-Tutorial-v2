import cv2
import random
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
import torch.utils.data as data

voc_class_indexs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
voc_class_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(voc_class_labels, range(len(voc_class_labels))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]

class VOCDataset(data.Dataset):
    def __init__(self, 
                 root   :str = None, 
                 image_set  = [('2007', 'trainval'), ('2012', 'trainval')],
                 is_train   :bool =False,
                 ):
        # ----------- Basic parameters -----------
        self.image_set = image_set
        self.is_train  = is_train
        self.num_classes = 20
        # ----------- Path parameters -----------
        self.root = root
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        # ----------- Data parameters -----------
        self.ids = list()
        for (year, name) in image_set:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        self.dataset_size = len(self.ids)
        self.class_labels = voc_class_labels
        self.class_indexs = voc_class_indexs
        # ----------- Transform parameters -----------
        self.target_transform = VOCAnnotationTransform()

    def __len__(self):
        return self.dataset_size

    def pull_item(self, index):
        # load an image
        img_id = self.ids[index]
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        height, width, channels = image.shape

        # laod an annotation
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self.target_transform(anno)

        # guard against no boxes via resizing
        anno = np.array(anno).reshape(-1, 5)
        bboxes = anno[:, :4]  # [N, 4]
        labels = anno[:, 4]   # [N,]
        target = {
            "file_name": "{}.jpg".format(img_id[-1]),
            "bboxes": bboxes,
            "labels": labels,
            "orig_size": [height, width],
            "id": index,
        }
        
        return target


if __name__ == "__main__":
    import json

    # json_file = "D:\\python_work\\dataset\\COCO\\annotations\\instances_val2017.json"
    # with open(json_file, 'r') as f:
    #     data_dict = json.load(f)
    # print(data_dict['info'])
    # print(data_dict.keys())
    # print(len(data_dict["annotations"]))
    # print(len(data_dict["images"]))
    # print(data_dict["images"][0])
    # print(data_dict["images"][1])
    # print(data_dict["images"][2])
    # print(data_dict["annotations"][0])
    # print(data_dict["annotations"][1])
    # print(data_dict["annotations"][2])
    # exit()

    # opt
    is_train = True
    dataset = VOCDataset(root='D:/python_work/dataset/VOCdevkit/',
                         image_set=[('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')],
                         is_train=is_train,
                         )
    
    print('Data length: ', len(dataset))

    coco_dict = {
        "images": [],
        "annotations": [],
        "type": "instances",
        "categories": [{'supercategory': name, "id": i, 'name': name} for i, name in enumerate(voc_class_labels)]
    }
    anno_id = 0
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(" - [{}] / [{}] ...".format(i, len(dataset)))

        target = dataset.pull_item(i)

        # images info.
        file_name = target["file_name"]
        height, width = target["orig_size"]
        id = int(target["id"])

        coco_dict["images"].append({
            'file_name': file_name,
            'height': height,
            'width': width,
            'id': id
        })

        # annotation info.
        bboxes = target["bboxes"]
        labels = target["labels"]

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            coco_dict["annotations"].append({
                'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                'area': int((x2 - x1) * (y2 - y1)),
                'category_id': int(label),
                'image_id': id,
                'id': anno_id,
                'iscrowd': 0,
            })
            anno_id += 1

    json_file = "D:\\python_work\\dataset\\VOCdevkit\\annotations\\instances_train.json"
    with open(json_file, 'w') as f:
        json.dump(coco_dict, f, indent=4)
    print(f"Data saved to {json_file}")
