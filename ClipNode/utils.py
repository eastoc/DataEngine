# Time: 2023.10.13
import json
import numpy as np
import socket

def write_json(database, file_name='dataset.json'):
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(database, file, indent=4)
        file.close()
    print('Build dataset.json: Done.')

def load_json(json_dir: str):
    with open(json_dir, 'rt', encoding='utf-8') as file:
        database = json.load(file)
        print("Successfully loading: %s."%(json_dir))
    return database

def cosin_sim(x, y):
    return 1-np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

def cosin_dist(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

def get_local_ip():
    # 获取本地主机名
    hostname = socket.gethostname()
    # 获取本地IP地址
    ip_address = socket.gethostbyname(hostname)
    return ip_address