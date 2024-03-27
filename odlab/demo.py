import cv2
import os
import time
import numpy as np
import imageio
import argparse
from PIL import Image

import torch

# load transform
from datasets import coco_labels, build_transform

# load some utils
from utils.misc import load_weight
from utils.vis_tools import visualize

from config import build_config
from models.detectors import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='General Object Detection Demo')
    # Basic
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='./dataset/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='dataset/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/demos/',
                        type=str, help='The path to save the detection results')
    parser.add_argument('-vt', '--visual_threshold', default=0.3, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show visualization')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')
    # Model
    parser.add_argument('-m', '--model', default='fcos_r18_1x', type=str,
                        help='build detector')
    parser.add_argument('-nc', '--num_classes', default=80, type=int,
                        help='number of classes.')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument("--deploy", action="store_true", default=False,
                        help="deploy mode or not")
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    return parser.parse_args()
                    

def detect(args, model, device, transform, class_names, class_colors):
    # path to save
    save_path = os.path.join(args.path_to_save, args.mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if args.mode == 'camera':
        print('use camera !!!')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                orig_h, orig_w, _ = frame.shape

                # to PIL
                image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

                # prepare
                x = transform(image)[0]
                x = x.unsqueeze(0).to(device)
                
                # Inference
                t0 = time.time()
                bboxes, scores, labels = model(x)
                print("Infer. time: {}".format(time.time() - t0, "s"))
                
                # Rescale bboxes
                bboxes[..., 0::2] *= orig_w
                bboxes[..., 1::2] *= orig_h

                # vis detection
                frame_vis = visualize(frame, bboxes, scores, labels, args.visual_threshold, class_colors, class_names)
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
    elif args.mode == 'video':
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

                # to PIL
                image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

                # prepare
                x = transform(image)[0]
                x = x.unsqueeze(0).to(device)
                
                # Inference
                t0 = time.time()
                bboxes, scores, labels = model(x)
                print("Infer. time: {}".format(time.time() - t0, "s"))
                
                # Rescale bboxes
                bboxes[..., 0::2] *= orig_w
                bboxes[..., 1::2] *= orig_h

                # vis detection
                frame_vis = visualize(frame, bboxes, scores, labels, args.visual_threshold, class_colors, class_names)
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
    elif args.mode == 'image':
        for i, img_id in enumerate(os.listdir(args.path_to_img)):
            cv2_image = cv2.imread((args.path_to_img + '/' + img_id), cv2.IMREAD_COLOR)
            orig_h, orig_w, _ = cv2_image.shape

            # to PIL
            image = Image.fromarray(cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGB))

            # prepare
            x = transform(image)[0]
            x = x.unsqueeze(0).to(device)
            
            # Inference
            t0 = time.time()
            bboxes, scores, labels = model(x)
            print("Infer. time: {}".format(time.time() - t0, "s"))
            
            # Rescale bboxes
            bboxes[..., 0::2] *= orig_w
            bboxes[..., 1::2] *= orig_h

            # vis detection
            img_processed = visualize(cv2_image, bboxes, scores, labels, args.visual_threshold, class_colors, class_names)
            cv2.imwrite(os.path.join(save_path, str(i).zfill(6)+'.jpg'), img_processed)
            if args.show:
                cv2.imshow('detection', img_processed)
                cv2.waitKey(0)


def run():
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Dataset & Model Config
    cfg = build_config(args)

    # Transform
    transform = build_transform(cfg, is_train=False)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255))
                     for _ in range(args.num_classes)]

    # Model
    model = build_model(args, cfg, device, args.num_classes, False)
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    print("================= DETECT =================")
    # run
    detect(args, model, device, transform, coco_labels, class_colors)


if __name__ == '__main__':
    run()
