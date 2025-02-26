import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# -------------------------- For Detection Task --------------------------
## visualize the input data during the training stage
def vis_data(images, targets, masks=None, class_labels=None, normalized_coord=False, box_format='xyxy'):
    """
        images: (tensor) [B, 3, H, W]
        masks: (Tensor) [B, H, W]
        targets: (list) a list of targets
    """
    batch_size = images.size(0)
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    pixel_means = [0.485, 0.456, 0.406]
    pixel_std   = [0.229, 0.224, 0.225]

    for bi in range(batch_size):
        target = targets[bi]
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        not_mask = ~masks[bi]
        img_h = not_mask.cumsum(0, dtype=torch.int32)[-1, 0]
        img_w = not_mask.cumsum(1, dtype=torch.int32)[0, -1]
        # denormalize
        image = (image * pixel_std + pixel_means) * 255
        image = image[:, :, (2, 1, 0)].astype(np.uint8)
        image = image.copy()

        tgt_boxes = target['boxes'].float()
        tgt_labels = target['labels'].long()
        for box, label in zip(tgt_boxes, tgt_labels):
            box_ = box.clone()
            if normalized_coord:
                box_[..., [0, 2]] *= img_w
                box_[..., [1, 3]] *= img_h
            if box_format == 'xywh':
                box_x1y1 = box_[..., :2] - box_[..., 2:] * 0.5
                box_x2y2 = box_[..., :2] + box_[..., 2:] * 0.5
                box_ = torch.cat([box_x1y1, box_x2y2], dim=-1)
            x1, y1, x2, y2 = box_.long().cpu().numpy()
            
            cls_id = label.item()
            color = class_colors[cls_id]
            # draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if class_labels is not None:
                class_name = class_labels[cls_id]
                # plot title bbox
                t_size = cv2.getTextSize(class_name, 0, fontScale=1, thickness=2)[0]
                cv2.rectangle(image, (x1, y1-t_size[1]), (int(x1 + t_size[0] * 0.4), y1), color, -1)
                # put the test on the title bbox
                cv2.putText(image, class_name, (x1, y1 - 5), 0, 0.4, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        cv2.imshow('train target', image)
        cv2.waitKey(0)

## Draw bbox & label on the image
def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img

## Visualize the detection results
def visualize(image, bboxes, scores, labels, class_colors, class_names):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        cls_id = int(labels[i])
        cls_color = class_colors[cls_id]
            
        mess = '%s: %.2f' % (class_names[cls_id], scores[i])
        image = plot_bbox_labels(image, bbox, mess, cls_color, text_scale=ts)

    return image