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

## plot bbox & label on image
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

## visualize detection
def visualize(img, 
              bboxes, 
              scores, 
              labels, 
              vis_thresh, 
              class_colors, 
              class_names):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(labels[i])
            cls_color = class_colors[cls_id]
                
            mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

## convert feature to he heatmap
def convert_feature_heatmap(feature):
    """
        feature: (ndarray) [H, W, C]
    """
    heatmap = None

    return heatmap

## draw feature on the image
def draw_feature(img, features, save=None):
    """
        img: (ndarray & cv2.Mat) [H, W, C], where the C is 3 for RGB or 1 for Gray.
        features: (List[ndarray]). It is a list of the multiple feature map whose shape is [H, W, C].
        save: (bool) save the result or not.
    """
    img_h, img_w = img.shape[:2]

    for i, fmp in enumerate(features):
        hmp = convert_feature_heatmap(fmp)
        hmp = cv2.resize(hmp, (img_w, img_h))
        hmp = hmp.astype(np.uint8)*255
        hmp_rgb = cv2.applyColorMap(hmp, cv2.COLORMAP_JET)
        
        superimposed_img = hmp_rgb * 0.4 + img 

        # show the heatmap
        plt.imshow(hmp)
        plt.close()

        # show the image with heatmap
        cv2.imshow("image with heatmap", superimposed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if save:
            save_dir = 'feature_heatmap'
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, 'feature_{}.png'.format(i) ), superimposed_img)    


# -------------------------- For Tracking Task --------------------------
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im
