import os
import json
from ClipNode.utils import write_json

if __name__=='__main__':
    json_name = 'SQL1.json'
    dir = '../Datasets/SQL/SQL1/'
    ImgFiles = os.listdir(dir)
    database = []
    for ImgFile in ImgFiles:
        ImgDir = os.path.join(dir, ImgFile)
        database.append(ImgDir)
    write_json(database, json_name)
    