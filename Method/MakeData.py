import numpy as np
import pandas as pd
import Method.Parts.PMakeData as p

#データを学習用に編集 基本形　左端のOpenからの相対値を所持する形
def make_data(start_point = 0,end_point = 1, look_bar_length = 100, get_length = 20, cut_length = 10, visible_length = 6, sudden = 10, pips = 0.0001, root = "/content/drive/MyDrive/ＦX予測/data/EURUSD_M5_1999_10_04_2022_03_16.csv"):
  data =p.load_data(root, start_point, end_point)#csv読み込み
  data = p.reshape_data(data, look_bar_length)#成形・過去データを横列に配置
  ans = p.make_ans(data,get_length, cut_length,visible_length, sudden, pips)#解作成　0か1
  data = p.relative_data(data,pips)[:len(ans)]#最新のOpenからの相対値に変換
  return data,ans

#データを学習用に編集　解を0～1でなく惜しい回答などに1以下の点を与える
def make_data_v2(start_point = 0,end_point = 1, look_bar_length = 100, get_length = 20, cut_length = 10, near_cut_score = [3.,1.5,1.,0.6], near_cut_correct=[2.,4.,6.,7.], visible_length = 6, sudden = 10, pips = 0.0001, root = "/content/drive/MyDrive/ＦX予測/data/EURUSD_M5_1999_10_04_2022_03_16.csv"):
  data =p.load_data(root, start_point, end_point)#読み込み
  data = p.reshape_data(data, look_bar_length)#成形・過去データ
  ans = p.make_ans_v2(data,get_length, cut_length, near_cut_score, near_cut_correct,visible_length, sudden, pips)#解作成　0～1で条件によって小数点以下の数値あり
  data = p.relative_data(data,pips)[:len(ans)]#Openからの相対値に変換
  return data,ans
#データを学習用に編集　各時間のOpenからの相対値に変換　Open枠はHighLowを設置
def make_data_v3(start_point = 0,end_point = 1, look_bar_length = 100, get_length = 20, cut_length = 10, visible_length = 6, sudden = 10, pips = 0.0001, root = "/content/drive/MyDrive/ＦX予測/data/EURUSD_M5_1999_10_04_2022_03_16.csv"):
  data =p.load_data(root, start_point, end_point)#csv読み込み
  data = p.reshape_data(data, look_bar_length)#成形・過去データを横列に配置
  ans = p.make_ans(data,get_length, cut_length,visible_length, sudden, pips)#解作成
  data = p.relative_data_v2(data,pips)[:len(ans)]#各時間のOpenからの相対値に変換　Open枠はHighLowを設置
  return data,ans