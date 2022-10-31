import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.functional import accuracy
input = 400
output = 7
class Net(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.BatchNorm1d(input),
        nn.Linear(input,300),
        nn.ReLU(),
        nn.BatchNorm1d(300),
        nn.Linear(300,200),
        nn.ReLU(),
        nn.BatchNorm1d(200),
        nn.Linear(200,100),
        nn.ReLU(),
        nn.Linear(100,output)
        )
  def forward(self,x):
    return self.model(x)
  def up_down_step(self, y, t,category="train"):
    weight = torch.tensor([1,1,1,1,1, 1, up_down_weight*0.7]).cuda()
    loss = F.binary_cross_entropy(y.softmax(dim=-1),t,weight)
    self.log(f'{category}_up_down', accuracy(y.softmax(dim=-1),t.to(torch.int64)), on_step=(category=="train"), on_epoch=True, prog_bar=(category=="train"))
    return loss
  def sudden_down_step(self, y,t,category="train"):
    weight = torch.tensor([1,down_weight,1,1,1, 1, 1]).cuda()
    loss = F.binary_cross_entropy(torch.sigmoid(y), t,weight)
    self.log(f'{category}_sudden_down', accuracy(y.softmax(dim=-1),t.to(torch.int64)), on_step=(category=="train"), on_epoch=True, prog_bar=(category=="train"))
    return loss
  def sudden_upp_step(self, y,t,category="train"):
    weight = torch.tensor([1,1,1,upp_weight,1, 1, 1]).cuda()
    loss = F.binary_cross_entropy(torch.sigmoid(y), t,weight)
    self.log(f'{category}_sudden_upp', accuracy(y.softmax(dim=-1),t.to(torch.int64)), on_step=(category=="train"), on_epoch=True, prog_bar=(category=="train"))
    return loss
  def step(self, batch, batch_idx, category="train"):
    x, t = batch
    y = self(x)
    loss = self.sudden_down_step(y[:2],t[:2],category)
    loss += self.sudden_upp_step(y[2:4],t[2:4], category) 
    loss += self.up_down_step(y[4:],t[4:],category)
    return loss
  def training_step(self, batch, batch_idx):
    return self.step(batch, batch_idx)
  # 検証データに対する処理
  def validation_step(self, batch, batch_idx):
    return self.step(batch,batch_idx, "val")
  # テストデータに対する処理
  def test_step(self, batch, batch_idx):
    return self.step(batch,batch_idx,"test")
  def configure_optimizers(self):
   optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
   return optimizer