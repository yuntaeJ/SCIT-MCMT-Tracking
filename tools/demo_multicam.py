import sys
import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np

from loguru import logger

sys.path.append('.')

# from yolox.data.data_augment import preproc
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
    
def make_parser():
    # mmdet detection args
    parser = argparse.ArgumentParser(description='Seg-SORT Demo')
    parser.add_argument('video', help='Video file path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    
    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.9, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.5, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.9, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=300, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')


    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def imageflow_demo(model, vis_folder, current_time, args):
    
    temp = []
    channel_list = [file for file in os.listdir(args.video)]
    for channel in channel_list:
        d = os.path.join(args.video, channel)
        if os.path.isdir(d):
            temp.append(channel)
    
    for channel in temp:
        video_path = os.path.join(os.path.join(args.video, channel, 'video.mp4'))
        
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        save_folder = osp.join(vis_folder, channel)
        os.makedirs(save_folder, exist_ok=True)
        # if args.demo == "video":
        save_path = osp.join(save_folder,"inference_vid.mp4")
        # else:
        #     save_path = osp.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

        ## Tracking Model
        tracker = BoTSORT(args, frame_rate=fps)
        timer = Timer()
        frame_id = 0
        results = []
        while True:
            if frame_id % 20 == 0:
                logger.info('Processing frame {} '.format(frame_id))
            ret_val, frame = cap.read()
            if ret_val:
                # Detect objects
                # outputs, img_info = predictor.inference(frame, timer)
                # scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))
                det_results = inference_detector(model, frame)

                # mmdet result to dets format [x1,y1,x2,y2,score,class_id]
                dets_list = []
                for class_id,class_reuslts in enumerate(det_results):
                    for dets in class_reuslts:
                        dets_list.append([dets[0],dets[1],dets[2],dets[3],dets[4],class_id])

                if dets_list is not None:
                    # outputs = outputs[0].cpu().numpy()
                    # detections = outputs[:, :7]
                    # detections[:, :4] /= scale

                    # Run tracker
                    online_targets = tracker.update(np.array(dets_list), frame)

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                    timer.toc()
                    online_im = plot_tracking(
                        frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )
                else:
                    timer.toc()
                    online_im = frame
                if args.out:
                    vid_writer.write(online_im)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1

        if args.out:
            res_file = osp.join(save_folder, f"{timestamp}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")
            vid_writer.release()


def main(args):
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    args.experiment_name = "seg-sort demo"

    output_dir = osp.join(args.out)
    os.makedirs(output_dir, exist_ok=True)

    if args.out:
        vis_folder = osp.join(args.out, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))
    
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    
    current_time = time.localtime()
    # if args.demo == "image" or args.demo == "images":
    #     image_demo(model, vis_folder, current_time, args)
    if args.video:
        imageflow_demo(model, vis_folder, current_time, args)
    else:
        raise ValueError("Error: No video")


if __name__ == "__main__":
    args = make_parser().parse_args()
    # exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(args)
