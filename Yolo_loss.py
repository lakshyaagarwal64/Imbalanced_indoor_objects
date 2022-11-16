#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.nn as nn
import utils


# In[12]:


class YoloLoss(nn.Module):
    def __init__(self,S=7,B=2,C=20):
        super(YoloLoss,self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        iou_b1 = utils.intersection_over_union(predictions[..., self.C+1:self.C+5], target[..., self.C+1:self.C+5])
        iou_b2 = utils.intersection_over_union(predictions[..., self.C+6:self.C+10], target[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[...,self.C].unsqueeze(3) #Iobj_i
        #For box coords
        box_predictions = exists_box*(
            (
               bestbox*predictions[..., self.C+6:self.C+10] + (1-bestbox)*predictions[...,self.C+1:self.C+5]
            )
        )
        
        box_targets = exists_box*target[...,self.C+1:self.C+5]
        
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[...,2:4]+ 1e-6))
        #(N,S,S,4) -> (N*S*S,4)
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])
        
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        
        #Object loss
        
        pred_box = (
            bestbox*predictions[...,self.C+5:self.C+6]+(1-bestbox)*predictions[...,self.C:self.C+1]
        )
        #(N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box*pred_box),
            torch.flatten(exists_box*target[...,self.C:self.C+1])
        )
        
        #No-object-loss
        #(N,S,S,1)--> (N,S*S)
        no_object_loss = self.mse(
            torch.flatten((1-exists_box)*predictions[...,self.C:self.C+1], start_dim=1),
            torch.flatten((1-exists_box)*target[...,self.C:self.C+1], start_dim=1)
        )
        
        no_object_loss += self.mse(
            torch.flatten((1-exists_box)*predictions[...,self.C+5:self.C+6], start_dim=1),
            torch.flatten((1-exists_box)*target[...,self.C:self.C+1], start_dim=1)
        )
        
        #Class-loss
        class_loss = self.mse(
            torch.flatten(exists_box*predictions[...,:self.C],end_dim=2),
            torch.flatten(exists_box*target[...,:self.C], end_dim=-2)
        )
        
        loss = (
            self.lambda_coord*box_loss
            +object_loss
            +self.lambda_noobj*no_object_loss
            +class_loss
        )
        
        return loss


# In[ ]:




