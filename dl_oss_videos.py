# -*- coding: utf-8 -*-
# @author: Jiang Wei
# @date: 2024/12/27
#**************************************************************************************
import os, argparse, glob, tqdm, shutil
import multiprocessing
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
        "--oss_url", type=str, default=None, help=""
    )
    parser.add_argument(
        "--scenario_set_dir", type=str, default=None, help=""
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help=""
    )
    
    # multi_processing
    parser.add_argument(
        "--n_proc", type=int, default=60, help="number of processing"
    )
    return parser.parse_args()


def parse_moment_ids(scenario_set_dir):
    """"""
    # collect xlsx files of given scenario set
    xlsx_paths = glob.glob(os.path.join(scenario_set_dir, "*.xlsx"))
    print(f"--- number of xlsx files: {len(xlsx_paths)}")
    # collect moment ids
    moment_ids = set()
    for xlsx_path in xlsx_paths:
        xlsx_name = os.path.basename(xlsx_path)
        
        data = pd.read_excel(xlsx_path) # pip install openpyxl

        print(f"--- {xlsx_name}: shape: {data.shape}")
        
        names = data.columns.tolist()
        if len(names) == 1:
            sub_moment_ids = data["id"].tolist()
        else:
            sub_moment_ids = data["moment.moment_id"].tolist()
        print(f"--- number of moments: {len(sub_moment_ids)}")
        
        moment_ids.update(sub_moment_ids)
    moment_ids = sorted(moment_ids)
    return moment_ids

def process(proc_ordinal, args, moment_ids):
    """"""
    # download videos
    for moment_id in tqdm.tqdm(moment_ids):
        oss_video_path = f"{args.oss_url}/{moment_id[:2]}/{moment_id}/raw/camera_front_standard_image_pkts.mp4" # with Forward Lidar
        sub_oss_video_path = f"{args.oss_url}/{moment_id[:2]}/{moment_id}/raw/camera_front_standard_sub_image_pkts.mp4" # without Forword Lidar
        # 
        save_video_dir = os.path.join(args.save_dir, moment_id)
        save_video_path = os.path.join(save_video_dir, "camera_front_standard_image_pkts.mp4")
        if not os.path.exists(save_video_path):
            cmd = f"ossutil cp -f {oss_video_path} {save_video_dir}/"
            status = os.system(cmd)
            if status != 0:
                if os.path.exists(save_video_dir):
                    shutil.rmtree(save_video_dir)
                print(f"ERROR: [ {cmd} ]: ossutil download failure!")
            if not os.path.exists(save_video_path):
                if os.path.exists(save_video_dir):
                    shutil.rmtree(save_video_dir)
                print(f"ERROR: [ {oss_video_path} ]: Withous Forword Lidar!")
    return
    
# main function  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main(args):
    """"""
    #----------------------------------------------------------------------------------
    # collect xlsx files of given scenario set
    moment_ids = parse_moment_ids(args.scenario_set_dir)

    print(f"--- number of total moment ids without repeat: {len(moment_ids)}")

    # initialize save directory
    scenario_set_name = os.path.basename(args.scenario_set_dir)
    args.save_dir = os.path.join(args.save_dir, scenario_set_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
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
