import numpy as np
import pandas as pd
OPEN=0
HIGH=1
LOW=2
CLOSE=3
SUDDEN_DOWN=0
SUDDEN_DOWN_NONE = 1
SUDDEN_UPP=2
SUDDEN_UPP_NONE = 3
DOWN=4
UPP=5
NONE=6

#データのロード
def load_data(root, start_point, end_point):
  data = np.loadtxt(fname=root,dtype=np.float32, delimiter='\t',skiprows=1,usecols=[2,3,4,5])#必要な列を切り出してnp化
  data = data[int(len(data)*start_point):int(len(data)*end_point)]#でかすぎるので読み込む範囲以外を切り捨て
  return data
#データを過去データを含むものに変更
def reshape_data(data, look_bar_length):
  data2 = np.zeros([len(data)-look_bar_length,4*look_bar_length])
  for i in range(len(data)-look_bar_length) :
    data2[i] = data[i:i+look_bar_length].reshape([1,4*look_bar_length])
  return data2
#データを現在からの相対値に変更
def relative_data(data,pips):
    for i in range(len(data)):
      data[i] -=data[i][0]
      data[i]/=pips
    return data
#データを各Openからの相対値に変更
def relative_data_v2(data,pips):
    for i in range(len(data)):
        data[i]/=pips#pips変換
        for j in range(int(len(data[i])/4)):#OpenHighLowClose
            data[i][j*4+HIGH] -=data[i][j*4+OPEN]#相対値に変換
            data[i][j*4+LOW] -=data[i][j*4+OPEN]#相対値に変換
            data[i][j*4+CLOSE] -=data[i][j*4+OPEN]#相対値に変換
            if data[i][j+CLOSE] > 0: #HighかLowかに変換
                data[i][j*4+OPEN] = 1
            else:
                data[i][j*4+OPEN] = 0
    return data

#急変の回答データを作成
def set_sudden_ans(data, ans, sudden = 10, visible_length = 3, pips=0.0001):
  for i in range(len(data)-2):
    now = data[i][CLOSE]
    low = min(data[i+1][LOW], data[i+2][LOW])
    high = max(data[i+1][HIGH],data[i+2][HIGH])
    if abs(now-low)/pips > sudden:
      ans[i][SUDDEN_DOWN] = 1
    else :
      ans[i][SUDDEN_DOWN_NONE]=1
    if abs(now-high)/pips > sudden:
      ans[i][SUDDEN_UPP] = 1
    else :
      ans[i][SUDDEN_UPP_NONE] = 1
#通常回答データの作成
def set_ans(data, ans, get_length = 20, cut_length = 8, visible_length = 60, pips = 0.0001):
  for i in range(len(data)-visible_length):
    now = data[i][CLOSE]
    high = data[i][CLOSE]
    low = data[i][CLOSE]
    ans[i][NONE]=1
    for j in range(1,visible_length):
      low = min(low, data[i+j][LOW])
      high = max(high,data[i+j][HIGH])
      if (abs(high-now)/pips < cut_length) & (abs(now-low)/pips >= get_length):
        ans[i][DOWN],ans[i][NONE] = 1,0
        break
      if (abs(now-low)/pips < cut_length) & (abs(high-now)/pips >= get_length):
        ans[i][UPP],ans[i][NONE] = 1,0
        break
#回答データの作成
def make_ans(data, get_length = 20, cut_length = 8, visible_length = 36, sudden = 10, pips = 0.0001):
  ans = np.zeros([len(data),7])
  set_sudden_ans(data,ans,sudden,visible_length,pips)
  set_ans(data,ans,get_length,cut_length, visible_length,pips)
  ans = ans[:-visible_length]
  return ans
  
#解値の補正
def correct_score(p, score, correct_line):
  for (s, line) in zip(score, correct_line):
    if p < line:
        return s
  return p
#改良版回答データの作成
def set_ans_v2(data, ans, get_length = 20, cut_length = 8, near_cut_score = [3. , 1.5, 1., 0.6], near_cut_correct=[2.,4.,6.,7.], visible_length = 60, pips = 0.0001):
  for i in range(len(data)-visible_length):
    now = data[i][CLOSE]
    high = data[i+1][HIGH]
    low=data[i+1][LOW]
    ans[i][NONE]=1
    for j in range(1,visible_length):
      low = min(low, data[i+j][LOW])
      high = max(high,data[i+j][HIGH])
      if (abs(now-high)/pips < (cut_length)) & (abs(now-low)/pips >= (get_length)):
        ans[i][DOWN],ans[i][NONE] = correct_score(high, near_cut_score, near_cut_correct) ,0
        break
      elif (abs(now-low)/pips < (cut_length)) & (abs(now-high)/pips >= (get_length)):
        ans[i][UPP],ans[i][NONE] = correct_score(low, near_cut_score, near_cut_correct) ,0
        break
#回答データの作成
def make_ans_v2(data, get_length = 20, cut_length = 8, near_cut_score = [3.,1.5,1.,0.6], near_cut_correct=[2.,4.,6.,7.], visible_length = 60, sudden = 10, pips = 0.0001):
  ans = np.zeros([len(data),7])
  set_sudden_ans(data,ans,sudden,visible_length,pips)
  set_ans_v2(data,ans,get_length,cut_length, near_cut_score, near_cut_correct, visible_length,pips)
  ans = ans[:-visible_length]
  return ans