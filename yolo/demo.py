import argparse
import cv2
import os
import time
import numpy as np
import imageio

import torch

# load transform
from dataset.build import build_transform

# load some utils
from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import visualize

from models import build_model
from config import build_config

from dataset.voc  import voc_class_labels
from dataset.coco import coco_class_labels
from yolo.dataset.custom import custom_class_labels


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
    # Basic setting
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='dataset/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='dataset/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/demos/',
                        type=str, help='The path to save the detection results')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show visualization')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # Model setting
    parser.add_argument('-m', '--model', default='yolo_n', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    # Data setting
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, custom.')

    return parser.parse_args()
                    

def detect(args,
           model, 
           device, 
           transform, 
           num_classes,
           class_names,
           mode='image'):
    # class color
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]
    save_path = os.path.join(args.path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        # 笔记本摄像头，index=0；外接摄像头，index=1；
        cap = cv2.VideoCapture(index=0, apiPreference=cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                orig_h, orig_w, _ = frame.shape

                # prepare
                x, _, ratio = transform(frame)
                x = x.unsqueeze(0).to(device)
                
                # inference
                t0 = time.time()
                outputs = model(x)
                scores = outputs['scores']
                labels = outputs['labels']
                bboxes = outputs['bboxes']
                t1 = time.time()
                print("Infer time: {:.1f} ms. ".format((t1 - t0) * 1000))

                # rescale bboxes
                bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)

                # vis detection
                frame_vis = visualize(image=frame, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      class_names=class_names
                                      )
                frame_resized = cv2.resize(frame_vis, save_size)
                out.write(frame_resized)

                if args.gif:
                    gif_resized = cv2.resize(frame, (640, 480))
                    gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                    image_list.append(gif_resized_rgb)

                if args.show:
                    cv2.imshow('detection', frame_resized)
                    cv2.waitKey(1)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # generate GIF
        if args.gif:
            save_gif_path =  os.path.join(save_path, 'gif_files')
            os.makedirs(save_gif_path, exist_ok=True)
            save_gif_name = os.path.join(save_gif_path, '{}.gif'.format(cur_time))
            print('generating GIF ...')
            imageio.mimsave(save_gif_name, image_list, fps=fps)
            print('GIF done: {}'.format(save_gif_name))

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(args.path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                orig_h, orig_w, _ = frame.shape

                # prepare
                x, _, ratio = transform(frame)
                x = x.unsqueeze(0).to(device)

                # inference
                t0 = time.time()
                outputs = model(x)
                scores = outputs['scores']
                labels = outputs['labels']
                bboxes = outputs['bboxes']
                t1 = time.time()
                print("Infer time: {:.1f} ms. ".format((t1 - t0) * 1000))

                # rescale bboxes
                bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)

                # vis detection
                frame_vis = visualize(image=frame, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      class_names=class_names
                                      )

                frame_resized = cv2.resize(frame_vis, save_size)
                out.write(frame_resized)

                if args.gif:
                    gif_resized = cv2.resize(frame, (640, 480))
                    gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                    image_list.append(gif_resized_rgb)

                if args.show:
                    cv2.imshow('detection', frame_resized)
                    cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()

        # generate GIF
        if args.gif:
            save_gif_path =  os.path.join(save_path, 'gif_files')
            os.makedirs(save_gif_path, exist_ok=True)
            save_gif_name = os.path.join(save_gif_path, '{}.gif'.format(cur_time))
            print('generating GIF ...')
            imageio.mimsave(save_gif_name, image_list, fps=fps)
            print('GIF done: {}'.format(save_gif_name))

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for i, img_id in enumerate(os.listdir(args.path_to_img)):
            image = cv2.imread((args.path_to_img + '/' + img_id), cv2.IMREAD_COLOR)
            orig_h, orig_w, _ = image.shape

            # prepare
            x, _, ratio = transform(image)
            x = x.unsqueeze(0).to(device)

            # inference
            t0 = time.time()
            outputs = model(x)
            scores = outputs['scores']
            labels = outputs['labels']
            bboxes = outputs['bboxes']
            t1 = time.time()
            print("Infer time: {:.1f} ms. ".format((t1 - t0) * 1000))

            # rescale bboxes
            bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)

            # vis detection
            img_processed = visualize(image=image, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      class_names=class_names
                                      )
            cv2.imwrite(os.path.join(save_path, str(i).zfill(6)+'.jpg'), img_processed)
            if args.show:
                cv2.imshow('detection', img_processed)
                cv2.waitKey(0)


def run():
    args = parse_args()
    # Dataset config
    if   args.dataset == "voc":
        cfg.num_classes = 20
        cfg.class_labels = voc_class_labels
    elif args.dataset == "coco":
        cfg.num_classes = 80
        cfg.class_labels = coco_class_labels
    elif args.dataset == "custom":
        cfg.num_classes = len(custom_class_labels)
        cfg.class_labels = custom_class_labels
    else:
        raise NotImplementedError("Unknown dataset: {}".format(args.dataset))
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Build config
    cfg = build_config(args)

    # Build model
    model = build_model(args, cfg, False)

    # Load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # Build transform
    transform = build_transform(cfg, is_train=False)

    print("================= DETECT =================")
    # Run demo
    detect(args         = args,
           mode         = args.mode,
           model        = model, 
           device       = device,
           transform    = transform,
           num_classes  = cfg.num_classes,
           class_names  = cfg.class_labels,
           )


if __name__ == '__main__':
    run()
