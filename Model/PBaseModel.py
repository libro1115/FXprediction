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
SUDDEN_DOWN=0
SUDDEN_DOWN_NONE = 1
SUDDEN_UPP=2
SUDDEN_UPP_NONE = 3
DOWN=4
UPP=5
NONE=6
#データのtensor化
def make_tensor_dataset(df,ans,dtype=[torch.float32, torch.float32],ratio=[0.6,0.2]):
  x=torch.tensor(df, dtype=dtype[0])
  t=torch.tensor(ans,dtype=dtype[1])
  dataset = torch.utils.data.TensorDataset(x,t)
  n_train = int(len(dataset)*ratio[0])
  n_val = int(len(dataset)*ratio[1])
  n_test = len(dataset)-n_train-n_val
  pl.seed_everything(0)
  train,val,test = torch.utils.data.random_split(dataset,[n_train,n_val,n_test])
  return train,val,test
#loader作成
def make_loader(train,val,test,batch_size=50):
  pl.seed_everything(0)
  train_loader=torch.utils.data.DataLoader(train, batch_size,shuffle=True, drop_last=True)
  val_loader=torch.utils.data.DataLoader(val, batch_size)
  test_loader=torch.utils.data.DataLoader(test, batch_size)
  return train_loader,val_loader,test_loader
#学習実行
def use_model(model,train_loader,val_loader,epoch=20,save_file=""):
  logger = CSVLogger(save_dir='logs', name='my_exp')
  trainer = pl.Trainer(max_epochs=epoch, gpus=1, deterministic=True, logger=logger)
  trainer.fit(model, train_loader, val_loader)
  if save_file != "":torch.save(model.state_dict(), save_file)
  return trainer
#グラフ描画
def draw_graph(name,num:int):
  log = pd.read_csv(f'logs/my_exp/version_{num}/metrics.csv')
  log[[f'train_{name}', 'epoch']].dropna(how='any', axis=0).reset_index()[f'train_{name}'].plot();
  log[[f'val_{name}', 'epoch']].dropna(how='any', axis=0).reset_index()[f'val_{name}'].plot();
#売り買いサインのスコア 
def sign_predict(predict,ans):
  soft = nn.Softmax(dim=0)
  y = torch.argmax(predict[0][DOWN:])+DOWN#数値調整
  t = torch.argmax(ans[DOWN:])+DOWN
  if y == t:#解一致
    if t == NONE:
      return 1
    else :
      return 2
  else:#解不一致
    if y == NONE:
      return -1
    else:
      return -2
#テスト
def score_test(test, model, win_point=18,loss_point=10, through_point=0):
  score = 0
  active_acc = 0
  active_loss = 0
  passive_acc = 0
  passive_loss = 0
  model.eval()
  for data in test:
    x,t = data
    y = model(x.unsqueeze(0))
    sign = sign_predict(y,t)
    if sign == 1:
      passive_acc += 1
    elif sign == 2:
      active_acc += 1
    elif sign == -1:
      passive_loss += 1
    else:
      active_loss += 1
  score = (active_acc*win_point) - (active_loss*loss_point) - (passive_loss * through_point)
  print("score:",score)
  print("active_acc:",active_acc)
  print("active_loss:",active_loss)
  print("passive_acc:",passive_acc)
  print("passive_loss:",passive_loss)
  
def IsSudden(t, type = "up"):
  if type == "up":
    return (t[SUDDEN_UPP] > t[SUDDEN_UPP_NONE])
  else:
    return (t[SUDDEN_DOWN] > t[SUDDEN_DOWN_NONE])
    
def sudden_predict(predict,ans,type="up"):
  y = IsSudden(predict, type)
  t = IsSudden(ans, type)
  if t & y:
    return 1
  elif t & (not y):
    return -1
  elif (not t) & y:
    return -2
  else:
    return 2
#急変化予測のスコア
def sudden_score(test, model, type="up"):
  model.eval()
  true_acc = 0
  true_loss = 0
  false_acc = 0
  false_loss = 0
  for data in test:
    x,t = data
    y = model(x.unsqueeze(0))[0]
    sign = sudden_predict(y,t, type)
    if sign == 1:
      true_acc += 1
    elif sign == 2:
      false_acc += 1
    elif sign == -1:
      true_loss += 1
    else:
      false_loss += 1
  print("true_accuracy:" , true_acc/(true_acc+true_loss))
  print("false_accuracy:" , false_acc/(false_acc+false_loss))
  print("y_true_accuracy" , true_acc/(true_acc+false_loss))
    
def a_step(model, y, t,weight, category=["train", "up_down"]):
  weight = torch.tensor(weight).cuda()
  loss = F.binary_cross_entropy(y.softmax(dim=-1),t,weight)
  model.log(f'{category[0]}_{category[1]}', accuracy(y.softmax(dim=-1),t.to(torch.bool).to(torch.int64)), on_step=(category=="train"), on_epoch=True, prog_bar=(category=="train"))
  return loss

def step(model, batch, batch_idx, category="train"):
  x, t = batch
  y = model(x)
  loss  = a_step(model, y[0:2],t[0:2], [1,model.down_weight,1,1,1,1,1], [category,"sudden_down"])
  loss += a_step(model, y[2:4],t[2:4], [1,1,1,model.upp_weight,1,1,1], [category,"sudden_upp"])
  loss += a_step(model, y[4: ],t[4: ],[1,1,1,1,1,1,model.up_down_weight*0.7], [category,"up_down"])
  return loss
