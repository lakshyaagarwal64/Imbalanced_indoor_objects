#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import os
import pandas as pd
from PIL import Image
from math import isnan


# In[6]:


labelmap = {}
labelmap['exit']=1
labelmap['fireextinguisher']=2
labelmap['chair']=3
labelmap['trashbin']=4
labelmap['screen']=5
labelmap['printer']=6
labelmap['clock']=0


# In[8]:


class VOCDataset_new(torch.utils.data.Dataset):
    def __init__(
        self, labelmap, csv_file, img_dir, S=7, B=2, C=7, transform=None, transformprime=None
    ):
        self.labelmap = labelmap
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.transformprime = transformprime
        self.S = S
        self.B = B
        self.C = C
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, index):
        labels = self.annotations.iloc[index].values
        boxes=[]
        for i in range(4):
            if i<2:
                y=5*i
            else:
                y=5*i+1
            classlabel=labels[y]
            if type(classlabel)==str:
                fakex,fakey = labels[y+2], labels[y+1]
                class_label, width, height = labels[y], labels[y+3], labels[y+4]
                x = (fakex+width/2)/1280
                y = (fakey+height/2)/720
                boxes.append([labelmap[class_label],x,y,width/1280,height/720])
        img_path = self.img_dir + labels[10]
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        
        if self.transform:
            image, boxes = self.transform(image, boxes)
        if self.transformprime:
            image, boxes = self.transformprime(image,boxes)
            
            
        label_matrix = torch.zeros((self.S, self.S, self.C+5*self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S*y), int(self.S*x)
            x_cell, y_cell = self.S*x-j, self.S*y-i
            width_cell, height_cell = (
                width*self.S,
                height*self.S,
            )
            
            if label_matrix[i,j,self.C]==0:
                label_matrix[i,j,self.C]=1
                box_coordinates = torch.tensor(
                    [x_cell,y_cell,width_cell,height_cell]
                )
                label_matrix[i,j,self.C+1:self.C+5]= box_coordinates
                label_matrix[i,j,class_label] = 1
        return image, label_matrix


# In[ ]:




