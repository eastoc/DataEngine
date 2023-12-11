from data_model import ImageTagRequest, ImageTagResponse
import os 
import base64
import uuid 
import urllib
import onnxruntime
import numpy as np
from PIL import Image
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from typing import List 
import socket

batch_size = 1

def convert_to_rgb(image):
    return image.convert("RGB")


def get_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_tag_list(tag_list_file):
        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

def get_local_ip():
    # 获取本地主机名
    hostname = socket.gethostname()
    # 获取本地IP地址
    ip_address = socket.gethostbyname(hostname)
    return ip_address
    
class RAM():
    def __init__(self):

        ip = get_local_ip()
        if ip == "127.0.1.1":
            self.ort_session = onnxruntime.InferenceSession("ram_swin_large_14m_b1.onnx", providers=["CPUExecutionProvider"])
        else:
            self.ort_session = onnxruntime.InferenceSession("/data/train/caption/ram_swin_large_14m_b1.onnx", providers=["CPUExecutionProvider"])
        print("IP: %s"%(ip))
        self.transform = get_transform()
        self.tag_list = load_tag_list("resources/ram_tag_list.txt")
        self.tag_list_chinese = load_tag_list("resources/ram_tag_list_chinese.txt")

    def inference_tags(self, image):
        #from time import time
        #t1 = time()
        
        image = Image.open(image_path)
        #t2 = time()
        result = self.inference_by_image_pil(image)
        #t3 = time()
        #print(t2-t1, t3-t2)
        
        return result

    def inference_by_image_pil(self, image):
        image_arrays = self.transform(image).unsqueeze(0).numpy()
        # compute ONNX Runtime output prediction
        ort_inputs = {self.ort_session.get_inputs()[0].name: image_arrays}
        ort_outs = self.ort_session.run(None, ort_inputs)
        index = np.argwhere(ort_outs[0][0] == 1)
        token = self.tag_list[index].squeeze(axis=1).tolist()
        # token_chinese = tag_list_chinese[index].squeeze(axis=1).tolist()
        return ImageTagResponse(tags=token)

def inference_tags_batch(req: List[dict]):
    available_images = []
    for r in req:
        image_url = r["url"]
        image_path = uuid.uuid4().hex + ".png"
        image_path, _ = urllib.request.urlretrieve(image_url, image_path)
        if image_path is None:
            continue
        r['image_path'] = image_path
        available_images.append(r)
    result = inference_by_images(available_images)
    for r in available_images:
        os.remove(r['image_path'])
    return result



def inference_tags_url(req: ImageTagRequest):
    #from time import time
    #t1 = time()
    image_url = req.image_url
    image_path = uuid.uuid4().hex + ".png"
    image_path, _ = urllib.request.urlretrieve(image_url, image_path)
    if image_path is None:
        return "Error: image_path is None"
    image = Image.open(image_path)
    #t2 = time()
    result = inference_by_image_pil(image)
    #t3 = time()
    #print(t2-t1, t3-t2)
    os.remove(image_path)
    return result


def inference_tags_base64(image_base64: str, suffix: str):
    imgdata = base64.b64decode(image_base64)
    image_path = uuid.uuid4().hex + "." + suffix
    file = open(image_path, 'wb')
    file.write(imgdata)
    file.close()
    if image_path is None:
        return "Error: image_path is None"
    image = Image.open(image_path)
    result = inference_by_image_pil(image)
    os.remove(image_path)
    return result


def inference_tags_local_file(image_path):
    import shutil   
    ext = os.path.splitext(image_path)[-1].lower()
    tmp_image_path = uuid.uuid4().hex + "." + ext
    shutil.copyfile(image_path, tmp_image_path)
    image = Image.open(tmp_image_path)
    result = inference_by_image_pil(image)
    os.remove(tmp_image_path)
    return result


def inference_by_images(all_images):
    result_dict = {}
    for image in all_images:
        image_pil = Image.open(image['image_path'])
        resp = inference_by_image_pil(image_pil)
        result_dict[image['id']] = ",".join(resp.tags)
    return result_dict


