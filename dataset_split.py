import os
import random
from PIL import Image
from tqdm import tqdm
from ClipNode.utils import write_json
import json

def random_sample(arr, n):
    return random.sample(arr, n)

def join(ParDir, SubDir):
    return os.path.join(ParDir, SubDir)

def split(category):
    # 给定指定类别，从train中划分出valid和test
    ratios = {"train":0.8,"valid":0.1,"test":0.1}
    TrainDir = f'data/filter/train/{category}'
    PosFiles = os.listdir(TrainDir)
    
    no_train_num = int(len(PosFiles) * (ratios['valid']+ratios['test']) )
    NoTrainFiles = random_sample(PosFiles, no_train_num)
    
    ValidNum = int(len(PosFiles) * ratios['valid'])
    TestNum = int(len(PosFiles) * ratios['test'])
    print(ValidNum, TestNum)
    ValidFiles  = random_sample(NoTrainFiles, ValidNum)
    TestFiles = list(set(NoTrainFiles) - set(ValidFiles))
    
    ValidDir = f"data/filter/valid/{category}/"
    TestDir = f"data/filter/test/{category}/"
    
    for ValidFile in ValidFiles:
        ValidImgDir = join(TrainDir, ValidFile)
        img = Image.open(ValidImgDir)
        img.save(join(ValidDir, ValidFile)) 
    write_json(ValidFiles, f'data/filter/valid_{category}.json')
    print(len(TestFiles))
    for TestFile in TestFiles:
        TestImgDir = join(TrainDir, TestFile)
        print(TestImgDir)
        img = Image.open(TestImgDir)
        img.save(join(TestDir, TestFile))
    write_json(TestFiles, f'data/filter/test_{category}.json')
    
def parse_json(json_file):
    name = json_file.split('.')[0]
    split, cate = name.split('_')
    return split, cate

def remove_train_from_json():
    dir = 'data/filter/'
    TrainDir = 'data/filter/train'
    parfiles = os.listdir(dir)
    for json_file in parfiles:
        if json_file.endswith('.json'):
            dataset, category = parse_json(json_file)
            with open(join(dir, json_file), 'r', encoding='utf-8') as f:
                ImgFiles = json.load(f)
            
            for ImgFile in ImgFiles:
                ImgDir = join(join(TrainDir, category), ImgFile)
                os.remove(ImgDir)
            
if __name__=='__main__':
    #split(category='trash')
    remove_train_from_json()
    