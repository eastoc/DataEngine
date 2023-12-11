import cv2
import os
import torchvision
from torchvision.models import ViT_L_32_Weights
from torchvision import transforms
import torch
import cv2
from torchvision import datasets
from torch.utils import data
from torchvision import models
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def wash_data():
    data_path = "/Users/east_wu/Downloads/707980003942597253/"
    data_files_path = os.listdir(data_path)
    format_list = []
    limit = 10000
    count = 1
    num = 1
    for i, file_path in enumerate(data_files_path):
        file_abspath = os.path.join(data_path, file_path)
        try:
            img = cv2.imread(file_abspath)
            [h, w, _] = img.shape
        except:
            os.remove(file_abspath)
            continue
        ratio = h / w
        if min(h, w) < 512:
            os.remove(file_abspath)
        elif ratio >= 4 or ratio <= 0.25:
            os.remove(file_abspath)

        format = file_path.split('.')[-1].lower()
        if format == 'gif':
            os.remove(file_abspath)
        elif format not in format_list:
            format_list.append(format)

        if count > limit:
            num += 1
            count = 1
        if format != 'jpg':
            file_path = file_path.split('.')[0] + '.jpg'
        save_path = os.path.join(str(num), file_path)
        cv2.imwrite(save_path, img)
        count += 1
    print(format_list)

class dataset():
    def __init__(self):
        self.data_dir = "bottle2"
        self.classes = os.listdir(self.data_dir)

        self.imgs = []
        self.gt = []
    def load_data(self, batch_size, train):

        transform = {
            'train': transforms.Compose(
                [transforms.Resize(512),
                 transforms.RandomCrop(512),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]),
            'test': transforms.Compose(
                [transforms.Resize(512),
                 transforms.RandomCrop(512),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
        }
        source_data = torchvision.datasets.ImageFolder(self.data_dir,
                                                       transform=transform['train' if train else 'test'])
        self.imgs = source_data.imgs
        self.gt = source_data.class_to_idx
        self.classes = source_data.classes
        data_loader = self.get_data_loader(source_data, batch_size=batch_size,
                                      shuffle=True if train else False,
                                      num_workers=1, drop_last=True if train else False)


        print('The size of source imdb is ', len(source_data))
        # PACS一个batch的数据是  227x227x3，
        return data_loader


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    """
    transform = ViT_L_32_Weights.IMAGENET1K_V1.transforms
    """
    transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder('data/filter/train', transform=transform)
    train_loader = data.DataLoader(train_data, batch_size=4, shuffle=True)
    valid_data = datasets.ImageFolder('data/filter/valid', transform=transform)
    valid_loader = data.DataLoader(train_data, batch_size=1, shuffle=True)
    model = models.vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
    #model = models.resnet18(pretrained=True)
    # 修改全连接层的输出
    #print(type())

    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, 2)
    
    model.to(device)

    criterion=torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    logger = SummaryWriter(log_dir='./log')
    epoches = 50
    losses = []
    for epoch in range(epoches):
        model.train()
        for i,data in enumerate(train_loader):
            
            (inputs, labels) = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            y_pred = model(inputs)
            loss=criterion(y_pred,labels)
            print(f"epoch: {epoch},loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward() #ReLU

            optimizer.step()
            losses.append(loss.to('cpu'))
        
        model.eval()
        tp = 0
        for i,data in enumerate(valid_loader):
            (inputs, labels) = data
            inputs = inputs.to(device)
            
            pred = model(inputs)
            pred = pred.detach().to('cpu').numpy()
            pred = np.argmax(pred[0])
            if pred == labels:
                tp += 1
        acc = tp/len(valid_loader)
        print(f"epoch {epoch}：accuracy={acc}")
        if acc >= max_acc:
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state,'ckpt/filter/best.pth')
            max_acc = acc
    