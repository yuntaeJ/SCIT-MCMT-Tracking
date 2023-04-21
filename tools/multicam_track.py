import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
from loguru import logger

# from yolox.data.data_augment import preproc
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.tp_tracker import TPTracker
from yolox.tracking_utils.timer import Timer

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

from model import SocialImplicit
from CFG import CFG
import numpy as np
from scipy.fftpack import fft, ifft

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("Trajectory-Prediction Track Demo!")
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    
    ########### mmdet bbox detector ##############
    parser.add_argument('video', help='Video file path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    ##############################################
    
    ########### trajectory predictor ##############
    parser.add_argument('--tp_weight', type=str, help='trajectory prediction model weight')
    ##############################################
    
#     parser.add_argument("-expn", "--experiment-name", type=str, default=None)
#     parser.add_argument("-n", "--name", type=str, default=None, help="model name")

#     parser.add_argument(
#         "--path", default="./videos/palace.mp4", help="channel level video path"
#     )
    # parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

    parser.add_argument('--save_result', type=str, help='Output video file')
    parser.add_argument('--save_vid', default=False, help='Output video file')
    
    # exp file
    # parser.add_argument(
    #     "-f",
    #     "--exp_file",
    #     default=None,
    #     type=str,
    #     help="pls input your expriment description file",
    # )
    # parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    # parser.add_argument(
    #     "--device",
    #     default="gpu",
    #     type=str,
    #     help="device to run our model, can either be cpu or gpu",
    # )
    # parser.add_argument("--conf", default=None, type=float, help="test conf")
    # parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    # parser.add_argument("--tsize", default=None, type=int, help="test img size")
    # parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    # parser.add_argument(
    #     "--fp16",
    #     dest="fp16",
    #     default=False,
    #     action="store_true",
    #     help="Adopting mix precision evaluating.",
    # )
    # parser.add_argument(
    #     "--fuse",
    #     dest="fuse",
    #     default=False,
    #     action="store_true",
    #     help="Fuse conv and bn for testing.",
    # )
    # parser.add_argument(
    #     "--trt",
    #     dest="trt",
    #     default=False,
    #     action="store_true",
    #     help="Using TensorRT model for testing.",
    # )
    
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.8, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    # parser.add_argument(
    #     "--aspect_ratio_thresh", type=float, default=1.6,
    #     help="threshold for filtering out boxes of which aspect ratio are above the given value."
    # )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    # parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
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



def imageflow_demo(det_model, vis_folder, current_time, args, tp_model):
    
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

        save_path = osp.join(save_folder,"inference_vid.mp4")

        logger.info(f"video save_path is {save_path}")
        if args.save_vid:
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
            )
        
        tracker = TPTracker(tp_model, args, frame_rate=30)
        timer = Timer()
        frame_id = 0
        results = []
        while True:
            if frame_id % 30 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            ret_val, frame = cap.read()
            if ret_val:
                
                timer.tic()
                det_results = inference_detector(det_model, frame)

                # mmdet result to dets format [x1,y1,x2,y2,score,class_id]
                dets_list = []
                for class_id,class_reuslts in enumerate(det_results):
                    for dets in class_reuslts:
                        dets_list.append([dets[0],dets[1],dets[2],dets[3],dets[4],class_id])
              
                if dets_list is not None:
                    # Run tracker
                    online_targets = tracker.update(np.array(dets_list), frame)
                    
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > args.min_box_area:
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
                if args.save_vid:
                    vid_writer.write(online_im)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1

        if args.save_result:
            res_file = osp.join(save_folder, f"{timestamp}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")

def main(args):

    output_dir = osp.join(args.save_result)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))
    
    det_model = init_detector(args.config, args.checkpoint, device=args.device)
    
    tp_model = SocialImplicit(spatial_input=CFG["spatial_input"],
                              spatial_output=CFG["spatial_output"],
                              temporal_input=CFG["temporal_input"],
                              temporal_output=CFG["temporal_output"],
                              bins=CFG["bins"],
                              noise_weight=CFG["noise_weight"]).cuda()
    tp_model.load_state_dict(torch.load(args.tp_weight))
    tp_model.cuda().double()
    tp_model.eval()
    
    current_time = time.localtime()
    if args.demo == "video":
        imageflow_demo(det_model, vis_folder, current_time, args, tp_model)

if __name__ == "__main__":
    args = make_parser().parse_args()
    # exp = get_exp(args.exp_file, args.name)
    
    main(args)
