import sys
import os
import argparse
import torch.nn as nn
import torch
from torchvision import models
from data_loader import get_dataloader
from loss_func import FocalLoss, NFLandRCE, NormalizedFocalLoss, NormalizedCrossEntropy
import torch.optim as optim
from misc import show_confMat, plot_line, cal_auc, Logger
from trainer import WarmupMultiStepLR, ModelTrainer
from datetime import datetime
import numpy as np
from models import MLPMixer
from checkpoint import save_checkpoint, takeFirst
import torch.nn.functional as F

# checkpoint1 = torch.load('./results/caogao/checkpoint_best_1st.pkl')
# c1 = checkpoint1['model_state_dict']
# checkpoint2 = torch.load('./results/caogao/checkpoint_best_2nd.pkl')
# c2 = checkpoint2['model_state_dict']
# checkpoint3 = torch.load('./results/caogao/checkpoint_best_3rd.pkl')
# c3 = checkpoint3['model_state_dict']
# print(checkpoint1['epoch'], checkpoint1['best_auc'], c1['fc.bias'])
# print(checkpoint2['epoch'],  checkpoint2['best_auc'], c2['fc.bias'])
# print(checkpoint3['epoch'],  checkpoint3['best_auc'],c3['fc.bias'])

loss = nn.CrossEntropyLoss()
input = torch.randn(8, 5)
print(input)
target = torch.empty(8,dtype=torch.long).random_(5)
print(target)
loss = loss(input, target)
print(loss)