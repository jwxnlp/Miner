# -*- coding: utf-8 -*-
# @author: Jiang Wei
# @date: 2024/12/27
#**************************************************************************************
import os, argparse, glob, tqdm, time
import multiprocessing

import numpy as np
import pandas as pd

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="Mine Corner Case using VLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #----------------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--scenario_set_image_dir", type=str, default=None, help=""
    )
    parser.add_argument(
        "--scenario_set_dir", type=str, default=None, help=""
    )
    parser.add_argument(
        "--txt_path", type=str, default=None, help=""
    )
    parser.add_argument(
        "--save_txt_dir", type=str, default=None, help=""
    )
    
    # multi_processing
    parser.add_argument(
        "--n_proc", type=int, default=32, help="number of processing"
    )
    parser.add_argument(
        "--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="dir of dataset"
    )
    return parser.parse_args()


def parse_moment_ids_from_txt(txt_path):
    """"""
    moment_ids = []
    with open(txt_path, "r") as f:
        for line in f:
            moment_id = line.strip()
            # check the format of moment's id
            if len(moment_id) == 36 and moment_id.count('-') == 4:
                moment_ids.append(moment_id)
            else:
                raise Exception("ERROR: [ {} ]: Novalid Format of Moment ID!".format(moment_id))
    return moment_ids


def get_messages(img_path, prompt):
    """"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        }
    ]
    return messages


def process(proc_ordinal, queue, gpu_id, args, moment_ids):
    """"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    
    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    # prompt for foggy
    #-----------------------------------------------------------------
    # prompt1 = "Is the weather shown in this image heavy foggy with low visibility? Please give a short anwser, like Yes/No!"
    # prompt2 = "Is the weather shown in this image heavy foggy with low visibility? Please give a short anwser, like Yes/No!"
    # prompt3 = "请仔细看看这张图片，展示的是在高速上驾驶的画面。细致地观察周围的环境，判断是否是大雾天气，即周围空中有很多水汽且能见度不到100米，请回答Yes或者No!"
    # prompt = "请仔细观察这张图片展示的驾驶环境，根据地面上的车道和交通标志等，判断是不是在高速上行驶，请回答Yes或者No!"
    
    # prompt for rain
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------
    condition = "1). 地面潮湿； 2). 地面积水或地面积水导致反光； 3). 自车前面玻璃上有水滴或者有雨刷在动； 4). 前面其他车辆疾驰时带起的水花或水汽； 5). 天空雨滴在落下，同时没有阳光且是阴天"
    prompt = f"请仔细看看这张图片，展示的是在高速上驾驶的画面。如果满足下面条件之一： {condition}，则是为雨天，否则不是。请细致地观察周围的环境， 并判断是否在是雨天，请回答Yes或者No!"
    
    consuming_time, N = 0, 0
    # download videos
    for moment_id in tqdm.tqdm(moment_ids):
        img_paths = sorted(glob.glob(os.path.join(
                args.scenario_set_image_dir, moment_id, "*.jpg")))
        img_paths = img_paths[::2]
        responses = []
        for img_path in img_paths:
            messages = get_messages(img_path, prompt)
            
            tic = time.time()
            
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            consuming_time += (time.time() - tic)
            N += 1
            
            # print(output_text)
            # convert ouput
            if "yes" in output_text[0].lower():
                responses.append(1)
            elif "no" in output_text[0].lower():
                responses.append(0)
            else:
                print(f"ERROR: [ {output_text[0]} ]: neither yes nor no in output_text")
            
        responses = np.array(responses)
        if responses.sum() / len(responses) >= 0.8:
            queue.put(moment_id)
    print("---Speed of Qwen2-VL-7B-Instruct Inference: [ {} ]mspf".format(consuming_time * 1000 / N))
    
    queue.put('end') # add indicator of processing end
    
    return
    
# main function  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main(args):
    """"""
    
    #----------------------------------------------------------------------------------
    if args.txt_path is not None:
        moment_ids = parse_moment_ids_from_txt(args.txt_path)
    else:
        moment_ids = sorted(os.listdir(args.scenario_set_image_dir))
    print(f"--- number of total moment ids without repeat: {len(moment_ids)}")

    # initialize save directory
    name = os.path.basename(args.scenario_set_image_dir)
    index = name.rfind("_")
    scenario_set_name = name[:index]
    args.scenario_set_recall_dir = os.path.join(os.path.dirname(args.scenario_set_image_dir), 
            f"{scenario_set_name}_recall")
    os.makedirs(args.scenario_set_recall_dir, exist_ok=True)
    
    # multiprocessing: consuming
    #----------------------------------------------------------------------
    gpu_ids = list(map(int, args.gpu_ids.split(","))) if args.gpu_ids != "-1" else []
    
    queue = multiprocessing.JoinableQueue()
    
    p_list = []
    for i in range(args.n_proc):
        split_moment_ids = moment_ids[i::args.n_proc]
        p = multiprocessing.Process(
            target=process,
            args=(i, queue, gpu_ids[i%len(gpu_ids)], args, split_moment_ids))
        p.start()
        print(f"ID of process p{i}: {p.pid}, {len(split_moment_ids)}")
        p_list.append(p)
    
    # consume
    #-------------------------------------------------------------------------
    # collect recalled moments
    indicators = []
    recall_moment_ids = []
    while len(indicators) != args.n_proc:
        res = queue.get()
        if res == 'end':
            indicators.append(res)
            continue
        recall_moment_ids.append(res)
        queue.task_done()
    
    print(f"--- number of recall moment ids: {len(recall_moment_ids)}")
    recall_moment_ids = sorted(list(set(recall_moment_ids)))
    print(f"--- number of successfully parsed moment ids after remove redundancy: {len(recall_moment_ids)}")
    
    # save
    #---------------------------------------------------------------------------------
    if len(recall_moment_ids) != 0:
        # timestamp = time.strftime("%Y%m%d", time.localtime())
        os.makedirs(args.save_txt_dir, exist_ok=True)
        save_txt_path = os.path.join(args.save_txt_dir, f"{scenario_set_name}.txt")
        if os.path.exists(save_txt_path):
            os.remove(save_txt_path)
        with open(save_txt_path, "w") as f:
            f.write("\n".join(recall_moment_ids))
    return


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    main(parse_args())