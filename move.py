import os
import random
import tqdm
random.seed(0)
ls = os.listdir('/data1/hjh/ProjectData/Defogging/OTS/haze/OTS')
random.shuffle(ls)
for input_file in tqdm.tqdm(ls[0:10000],desc=f'正在保存:'):
    # os.system(f"echo 'cp /data1/hjh/ProjectData/Defogging/OTS/haze/OTS{input_file} ~/Project/DerainNet/JackData/OTS/haze' ")
    os.system(f"rm /data1/hjh/ProjectData/Defogging/OTS/haze/OTS/{input_file} ~/Project/DerainNet/JackData/OTS/haze")