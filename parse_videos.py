# -*- coding: utf-8 -*-
# @author: Jiang Wei
# @date: 2024/12/27
#**************************************************************************************
import os, argparse, glob, tqdm, cv2
import multiprocessing

import numpy as np
import pandas as pd

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="Mine Corner Case using VLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #----------------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--scenario_set_video_dir", type=str, default=None, help=""
    )
    parser.add_argument(
        "--scenario_set_dir", type=str, default=None, help=""
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help=""
    )
    
    # multi_processing
    parser.add_argument(
        "--n_proc", type=int, default=60, help="number of processing"
    )
    return parser.parse_args()


def process(proc_ordinal, args, moment_ids):
    """"""
    # download videos
    for moment_id in tqdm.tqdm(moment_ids):
        video_path = os.path.join(args.scenario_set_video_dir, moment_id, 
                "camera_front_standard_image_pkts.mp4")
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened() == False:
            raise Exception("INFO: [ {} ]: Fail to Open!".format(video_path))
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print("--- frame height: {}, frame width: {}".format(width, height))
        # print("--- number of frames: {}".format(N))
        # print("--- fps: {}".format(fps))
        
        save_img_dir = os.path.join(args.scenario_set_image_dir, moment_id)
        os.makedirs(save_img_dir, exist_ok=True)
        if args.num_samples >= N:
            indices = np.arange(N)
        else:
            # indices = np.random.choice(np.arange(N), size=args.num_samples, replace=False)
            indices = np.linspace(0, N, args.num_samples)
            indices = np.around(indices).astype(np.int32)
        for idx in range(N):
            ret, frame = cap.read()
            if ret:
                # subsample
                #--------------------------------------------------------------
                if idx in indices:
                    save_img_path = os.path.join(save_img_dir, f"{idx}.jpg")
                    resized_frame = cv2.resize(frame, (width//2, height//2))
                    cv2.imwrite(save_img_path, resized_frame)
                
            else:
                # raise Exception("INFO: [ {} ]: Fail to Read [ {} ]th Frame!".format(video_path, idx))
                print("INFO: [ {} ]: Fail to Read [ {} ]th Frame!".format(video_path, idx))
        cap.release()
    return
    
# main function  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main(args):
    """"""
    #----------------------------------------------------------------------------------
    moment_ids = sorted(os.listdir(args.scenario_set_video_dir))
    print(f"--- number of total moment ids without repeat: {len(moment_ids)}")

    # initialize save directory
    args.scenario_set_image_dir = f"{args.scenario_set_video_dir}_images"
    os.makedirs(args.scenario_set_image_dir, exist_ok=True)
    
    # multiprocessing: consuming
    #----------------------------------------------------------------------
    p_list = []
    for i in range(args.n_proc):
        split_moment_ids = moment_ids[i::args.n_proc]
        p = multiprocessing.Process(
            target=process,
            args=(i, args, split_moment_ids))
        p.start()
        print(f"ID of process p{i}: {p.pid}, {len(split_moment_ids)}")
        p_list.append(p)

    for p in p_list:
        p.join()
        
    return

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    main(parse_args())
