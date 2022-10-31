import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
#データのtensor化
def make_tensor_dataset(df,ans,dtype=[torch.float32, torch.float32],ratio=[0.6,0.2]):
  x=torch.tensor(df, dtype=dtype[0])
  t=torch.tensor(ans,dtype=dtype[1])
  dataset = torch.utils.data.TensorDataset(x,t)
  n_train = int(len(dataset)*ratio[0])
  n_val = int(len(dataset)*ratio[1])
  n_test = len(dataset)-n_train-n_val
  #pl.seed_everything(0)
  train,val,test = torch.utils.data.random_split(dataset,[n_train,n_val,n_test])
  return train,val,test
#loader作成
def make_loader(train,val,test,batch_size=50):
  #pl.seed_everything(0)
  train_loader=torch.utils.data.DataLoader(train, batch_size,shuffle=True, drop_last=True)
  val_loader=torch.utils.data.DataLoader(val, batch_size)
  test_loader=torch.utils.data.DataLoader(test, batch_size)
  return train_loader,val_loader,test_loader
# #学習実行
# def use_model(model,train_loader,val_loader,epoch=20,save_file=""):
#   logger = CSVLogger(save_dir='logs', name='my_exp')
#   trainer = pl.Trainer(max_epochs=epoch, gpus=1, deterministic=True, logger=logger)
#   trainer.fit(model, train_loader, val_loader)
#   if save_file != "":torch.save(model.state_dict(), save_file)
#   return trainer
# #グラフ描画
# def draw_graph(name,num:int):
#   log = pd.read_csv(f'logs/my_exp/version_{num}/metrics.csv')
#   log[[f'train_{name}_epoch', 'epoch']].dropna(how='any', axis=0).reset_index()[f'train_{name}_epoch'].plot();
#   log[[f'val_{name}', 'epoch']].dropna(how='any', axis=0).reset_index()[f'val_{name}'].plot();
# #テスト
# def score_test(test, model, win_point=18,loss_point=10, through_point=0):
#   score = 0
#   active_acc = 0
#   active_loss = 0
#   passive_acc = 0
#   passive_loss = 0
#   model.eval()
#   for data in test:
#     x,t = data
#     y = torch.argmax(torch.sigmoid(model(x.unsqueeze(0))[0][DOWN:]))
#     t = torch.argmax(t[DOWN:])
#     if y == t:
#       if t == 2:
#         passive_acc +=1
#       else :
#         score += win_point
#         active_acc += 1
#     else:
#       if y == 2:
#         score -=through_point
#         passive_loss +=1
#       else:
#         score -= loss_point
#         active_loss += 1
#   print("score",score)
#   print("active_acc",active_acc)
#   print("active_loss",active_loss)
#   print("passive_acc",passive_acc)
#   print("passive_loss",passive_loss)