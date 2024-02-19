import torch
import torchvision 
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os
import PIL
from torch import nn
import torch.nn.functional as F
import PIL.Image
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import transforms
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LeakyReLU

from torch.nn import LogSoftmax
from torch import flatten
import matplotlib.pyplot as plt
import torch
import torchvision 
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os
import PIL
from torch import nn
import torch.nn.functional as F
import PIL.Image
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import transforms
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from IPython.display import display

class myDataset(Dataset):
    def __init__(self, csv , path ,transform=None):
        self.labels = pd.read_csv(csv)
        self.transform = transform
#         self.path = 'train/train_new/'
        self.path = path
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.path , self.labels.iloc[idx,0])
        image = PIL.Image.open(img_name)
        label = torch.tensor(self.labels.iloc[idx,1])
        if (self.transform):
            image = self.transform(image)
        return image, label


class model_mn(nn.Module):
    #convolution Block
    def conv_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    #mobileNet Block
    def convDw_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            #dw
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            #pw
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
   
    def __init__(self, conv_count=4 , mn_count=4):
        super(model_mn, self).__init__()
        #input channels
        self.x = 1
        #output channels
        self.y = 32
        #stride
        self.s = 2
        self.model = nn.Sequential()
        for i in range(conv_count):
            self.model.add_module("conv"+str(i+1), self.conv_block(self.x, self.y, self.s))
            self.x = self.y
            self.y = self.y*2
        
#         for i in range(mn_count):
#             if i%2 ==0:
#                 self.s = 2 
#             else:
#                 self.s = 1
#             self.model.add_module("convDw"+str(i+1), self.convDw_block(self.x, self.y, self.s))
#             self.x = self.y
#             self.y = self.y*2
        
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Linear(256*2*2 , 10)
        
        
    def forward(self, x):
        x = self.model(x)
#         print(x.shape)
        x = x.view(x.size(0), -1)
#         print(x.shape)
        x = self.fc(x)
        act = nn.Softmax(dim=1)
        x = act(x)
        return x
    
    def train_(self,train_data , epochs=6, lr=0.01 , optimizer='sgd'):
        train , val = torch.utils.data.random_split(train_data, [int(len(train_data)*0.8) , int(len(train_data)*0.2)], generator=torch.Generator().manual_seed(42))
        train_loader = torch.utils.data.DataLoader(train, batch_size=20, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=20, shuffle=True)

        error=nn.CrossEntropyLoss()
        if ( optimizer =='sgd'):
            optimizer=torch.optim.SGD(self.parameters(), lr)
        else:
            print('here')
            optimizer=torch.optim.Adam(self.parameters(), lr)
        epoch_acc = []
        train_acc = []
        epoch_loss= []
        train_loss = []
        val_acc = []
        val_loss = []
        va =[]
        vl = []
        prev_val_loss = 1
        for i in range(epochs):
            total_loss = 0
            va_loss = 0
            for x,y in (train_loader):
                xs, ys = x.to(device) , y.to(device)
#                 print(xs.shape)
#                 target = torch.argmax(ys, dim=-1)
                ypred = self.forward(xs)
#                 _, pre =  torch.max(ypred.data, 1)
                pre = torch.argmax(ypred, dim=1)
#                 print(pre , ys)
                loss = error(ypred, ys)
                epoch_loss.append(loss.detach().numpy())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss
                epoch_acc.append((pre == ys).sum().item() / pre.size(0))
            total_loss /= len(train_loader)
            train_loss.append(total_loss.detach().numpy())
            acc = np.mean(epoch_acc)
            train_acc.append(acc)
            print(f"EPOCH {i} TRAINING LOSS : {total_loss:.4f}")
            print(f"EPOCH {i} TRAINING ACCURACY: {acc:.4f}")
            
    
            for xv,yv in (val_loader):
                xsv, ysv = xv.to(device) , yv.to(device)
                ypredv = self.forward(xsv)
                _, prev =  torch.max(ypredv.data, 1)
                lossv = error(ypredv, ysv)
                val_loss.append(lossv.detach().numpy())
                lossv.backward()
                optimizer.step()
                optimizer.zero_grad()
                va_loss += lossv
                val_acc.append((prev == ysv).sum().item() / prev.size(0))
            va_loss /= len(val_loader)
            vl.append(va_loss.detach().numpy())
            va_acc = np.mean(val_acc)
            va.append(va_acc)
            print(f"EPOCH {i} VALIDATION LOSS : {va_loss:.4f}")
            print(f"EPOCH {i} VALIDATION ACCURACY: {va_acc:.4f}")
         
        #early stopping
#             if((va_loss.detach().numpy()) <= prev_val_loss ):
#                 prev_val_loss = va_loss.detach().numpy()
#                 continue
#             else:
#                 break
        plt.plot(train_loss, 'r' , label = "Loss") # plotting t, a separately 
        plt.plot(train_acc, 'b' , label = "Accuracy") # plotting t, b separately 
        plt.xlabel("Epochs")
        plt.title('Loss and Accuracy Curves of Training Data')
        plt.legend()
        plt.show()
            
        plt.plot(vl, 'r' , label = "Loss") # plotting t, a separately 
        plt.plot(va, 'b' , label = "Accuracy") # plotting t, b separately 
        plt.xlabel("Epochs")
        plt.title('Loss and Accuracy Curves of Validation Data')
        plt.legend()
        plt.show()    
        
    def test_(self, test_data):
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)
        a = []
        pr = []
        lb = []
        for img, label in test_loader:
            img , lab = img.to(device) , label.to(device)
            out = (self.forward(img)).round()
            _, predicted = torch.max(out.data, 1)
#             pr.append(predicted.detach().numpy())
#             _, l = torch.max(label.data, -1)
#             lb.append(l.detach().numpy())
        #     print(predicted.shape)
            a.append((predicted == label).sum().item() / predicted.size(0))
#             print(np.argmax(predicted.detach().numpy()), l)
            pr.extend(list(predicted.detach().numpy()))
            lb.extend(list(label.detach().numpy()))
        
        for i in range(len(img)):
            plt.imshow(img[i][0])
            plt.show()
            print("Predicted:" , predicted.detach().numpy()[i])
            print("Ground Truth:" , label.detach().numpy()[i])
        
#         print(predicted.detach().numpy(), label.detach().numpy())
        print(len(img))
        df = pd.DataFrame(confusion_matrix(lb , pr , labels=np.arange(0,10)))
        display(df)
        print("RECALL:" , recall_score(lb , pr , average='macro' ))
        print("TEST ACCURACY" , np.mean(a))
        
        
def main():
    print('run')
    net = model_mn().to(device)
    
    transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize(0,0.5)])
#     transform=transforms.Compose([
# #                                transforms.Resize(image_size),
# #                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                transforms.RandomHorizontalFlip(p=1),
#                                transforms.RandomRotation((90,90),expand=False, center=None, fill=0, resample=None),
                               
#                               ])
    
    train_data = myDataset('train/train.csv' , 'train/train_new' ,transform )
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True)
#     plt.imshow(train_data[0][0][0] , cmap='gray')

    test_data = myDataset('test/test.csv', 'test/test_new' ,transform )
#     plt.imshow(test_data[0][0][0] , cmap='gray')

    net.train_(train_data )
    
    
    
    
#     torch.save(net, 'model_gap.pth')
# We can then load the model like this:

#     model_ = torch.load('model_gap.pth')
#     model_.eval() # to turn off dropout
    net.test_(test_data)
    
#     for name, layer in model_.named_modules():
#         if isinstance(layer, torch.nn.Conv2d):
#             print(name, layer)
    
    
    
#     print(model_.model.conv4[0].weight.detach().clone())
    
  
        
    print_filters = True
    if (print_filters==True):
        m = torch.load('model_gap.pth')
        

        f0 = m.model.conv1[0].weight.detach().clone()
        f0 = f0 - f0.min()
        f0 = f0 / f0.max()
        print("First Conv Layer's Filters Size:" , m.model.conv1[0].weight.detach().clone().size())

        filter_img = torchvision.utils.make_grid(f0, nrow = 16 , normalize=True, padding=1)
        plt.figure( figsize=(100,500) )
        plt.imshow(filter_img.numpy().transpose((1, 2, 0)))
        plt.show()
   
        f = m.model.conv4[0].weight.detach().clone()
        n,c,w,h = f.shape
        f = f.view(n*c, -1, w, h)
        print("Last Conv Layer's Filters Size:" , m.model.conv4[0].weight.detach().clone().size())
        f = f - f.min()
        f = f / f.max()
        filter_img = torchvision.utils.make_grid(f, nrow = 32 , normalize=True, padding=1)
        plt.figure( figsize=(500, 500) )
        print(f.shape)
        print(filter_img.numpy().transpose((1, 2, 0)).shape)
        plt.imshow(filter_img.numpy().transpose((1, 2, 0)))
        plt.show()

     
    



        
if __name__ == '__main__':
    main()
