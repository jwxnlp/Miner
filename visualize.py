# -*- coding: utf-8 -*-
# @author: Jiang Wei
# @date: 2024/12/16
#**************************************************************************************
import os, argparse, cv2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="Mine Corner Case using VLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #----------------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--img_path", type=str, default=None, help=""
    )
    
    return parser.parse_args()
  
    
# main function  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main(args):
    """"""
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"{args.img_path}")
    img = cv2.imread(args.img_path)
    if img is None:
        raise Exception(f"ERROR: [ {args.img_path} ]: Not Exist!")
    
    bboxes = [
        ("dog", (100, 100), (500, 500)),
        ("Person", (450, 350), (600, 500))
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for bbox in bboxes:
        class_name, (x1, y1), (x2, y2) = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
        cv2.putText(
            img, f"{class_name}", 
            (int(x1), int(y1)-3), 
            font, 1, (0,0,255), thickness=1
        )
    cv2.imwrite(r"demo.png", img)
    return

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    main(parse_args())
