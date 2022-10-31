import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import Model.PBaseModel as p
import os

save_dir = "/content/drive/MyDrive/ＦX予測/Model/"

class BaseModel(pl.LightningModule):
  def __init__(self, data, ans, model, root="", batch_size = 50, dtype = [torch.float32, torch.float32], split_ratio = [0.6,0.2]):
    super().__init__()
    self.root =root + (f"_d{int(data.shape[1]/4)}.pth")
    self.d=dtype
    self.down_weight = ans.T[p.SUDDEN_DOWN].sum()/len(ans) 
    self.upp_weight = ans.T[p.SUDDEN_UPP].sum()/len(ans)
    self.up_down_weight = (len(ans)-ans.T[p.NONE].sum())/len(ans)
    self.input = data.shape[1]
    self.output = ans.shape[1]
    train,val,self.test = p.make_tensor_dataset(data, ans, dtype, split_ratio)
    self.train_loader,self.val_loader,self.test_loader = p.make_loader(train,val,self.test,batch_size)

    self.model = model
    if root != "":
      self.load_model()
    
  def forward(self,x):
    return self.model(x)
  # 学習データに対する処理
  def training_step(self, batch, batch_idx):
    return p.step(self,batch, batch_idx)
  # 検証データに対する処理
  def validation_step(self, batch, batch_idx):
    return p.step(self,batch,batch_idx, "val")
  # テストデータに対する処理
  def test_step(self, batch, batch_idx):
    return p.step(self,batch,batch_idx,"test")
  #学習実行
  def train_model(self,epock):
    return p.use_model(self, self.train_loader,self.val_loader,epock)
  #スコア表示
  def draw_score(self,win_point=18,loss_point=10, through_point=0):
    p.score_test(self.test, self, win_point,loss_point, through_point)
  def draw_score2(self,df,ans,win_point=18,loss_point=10, through_point=0):
    test = p.make_tensor_dataset(df,ans,self.d,ratio=[1,0])[0]
    p.score_test(test, self, win_point,loss_point, through_point)
  def sudden_score(self, type="up"):
    p.sudden_score(self.test, self, type)
  def sudden_score2(self,df,ans, type="up"):
    test = p.make_tensor_dataset(df,ans,self.d,ratio=[1,0])[0]
    p.sudden_score(test, self, type)
  
  #グラフ表示
  def graph(self,num:int,type_=0):
    lib = ["up_down","sudden_down","sudden_upp"]
    p.draw_graph(lib[type_], num)
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
    return optimizer
  #読み込み
  def load_model(self,root=""):
    if root == "":
      root = save_dir + self.root
    if os.path.isfile(root):
      self.load_state_dict(torch.load(root))
      print("load complete")
    else :
        print("not load")
  #上書き保存
  def save_model(self,root=""):
    if root == "":
      root = self.root
    torch.save(self.state_dict(),save_dir + root)
    device = torch.device("cpu")
    self.eval()
    traced_net = torch.jit.trace(self.forward, torch.rand(1,self.input).to(device))
    traced_net.save(save_dir + "C++" + root)