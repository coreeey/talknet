import pandas
import os
import tqdm
import numpy
import cv2
import glob
import argparse
import shutil
from tqdm import tqdm
import subprocess
import numpy as np
import random
import sys, time
from scipy.io import wavfile
import json
import csv
# trialFileName = r'/home/tanxxx/桌面/debuging/Talkies/talkies_val.csv'
def gen_ave_type():
    trialFileName = r'/mnt/lustre02/jiangsu/aispeech/home/flc23/DataSets/Talkies/talkies_train.csv'
    dataPath = r'/mnt/lustre02/jiangsu/aispeech/home/flc23/DataSets/Talkies'
    mix_lst = csv.reader(open(trialFileName))
    info_lst = []
    for i, item in enumerate(tqdm(mix_lst)):
        # if i > 2:
        #     break
        if i == 0:
            head = item.copy()
            head.extend(["timestamp", "id", "x1", "y1", "x2", "y2"])
            # info_lst.append(item)
            continue
        # face_id_ls = []
        # if item[0] not in ["8024c222-071c-44c0-af22-25b035961c69", "176213fd-1cb0-4745-992b-a0d925509257", "a20c1b37-e2d9-4f25-8324-d3dbe968b47f"]:
        #     continue
        meta_json = json.load(open(dataPath + '/' + 'meta/' + item[2].split('/')[-1], 'r', encoding="utf-8"))
        for j, info in enumerate(meta_json['faces']):
            info_mid = item.copy()  # keep info in one list
            info_mid.extend([info["timestamp"], info["id"], info["x1"], info["y1"], info["x2"], info["y2"]])
            info_lst.append(info_mid)

    f = open('talkies_train_check.csv', 'w')
    # write = csv.writer(f)
    write = csv.DictWriter(f, fieldnames=head)
    # 写入表头
    write.writeheader()

    for data in info_lst:
        print(data)
        data_ = {head[0]: data[0], head[1]: data[1], head[2]: data[2], head[3]: data[3], head[4]: data[4],
                 head[5]: data[5], head[6]: data[6], head[7]: data[7], head[8]: data[8], head[9]: data[9], head[10]: data[10]}
        write.writerow(data_)

def gen_