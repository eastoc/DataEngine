# Time: 2023.10.12
import torch
import clip
from PIL import Image
import os
import time
from .utils import get_local_ip
from tqdm import tqdm
import numpy as np

class encoder():
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print('Model is running with',self.device)       
        self.model, self.preprocess = clip.load("ClipNode/ViT-B-32.pt", device=self.device)
            
    def extract_feat(self, img):
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(image).to('cpu').detach().numpy()
        return feat

    def extract_text_feat(self, text: list):
        text = clip.tokenize(text).to(self.device)
        #print(text.size())
        with torch.no_grad():
            text_feat = self.model.encode_text(text).to('cpu').detach().numpy()
        return text_feat
         
    def build_dataset(self, imgs_dir):
        imgs_file = os.listdir(imgs_dir)
        start = time.time()
        database = []
        for i, img_file in tqdm(enumerate(imgs_file)):
            data = dict()
            img_dir = os.path.join(imgs_dir, img_file)
            try:
                img = Image.open(img_dir).convert("RGB")
                vec = self.extract_feat(img)[0]
                vec = vec/np.linalg.norm(vec)
                data['embedding'] = vec.tolist()
                data['image'] = img_file
                database.append(data)
            except:
                print(img_dir)
                #os.remove(img_dir)
        end = time.time()
        duration = (end - start)
        print("time cost: ", duration)
        
        return database


#text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

#with torch.no_grad():
    #image_features = model.encode_image(image)
    #text_features = model.encode_text(text)
    
    #logits_per_image, logits_per_text = model(image, text)
    #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
#print("image embeddings: ", image_features)
if __name__=='__main__':
    
    model = encoder()
    text = ['I would like beef noodles on the table, which can be decorated with spices such as plants and garlic. The color bands are shorter and have a sense of layering. The table is by the window and should be as simple and elegant as possible']
    feat = model.extract_text_feat(text)
    
    