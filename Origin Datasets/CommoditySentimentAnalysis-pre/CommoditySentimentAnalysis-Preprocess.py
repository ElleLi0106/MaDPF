import importlib
import os
import numpy as np
import pandas as pd
import sys

sys.path.append("../")
tools = importlib.import_module("data_preprocessing_tools")

dataPath_camera_label = r"F:\噪声数据处理\实验\Datasets\Sentiment Analysis 中文情感分析\\camera_label.txt"
dataPath_camera_sentence = r"F:\噪声数据处理\实验\Datasets\Sentiment Analysis 中文情感分析\\camera_sentence.txt"
dataPath_car_label = r"F:\噪声数据处理\实验\Datasets\Sentiment Analysis 中文情感分析\\car_label.txt"
dataPath_car_sentence = r"F:\噪声数据处理\实验\Datasets\Sentiment Analysis 中文情感分析\\car_sentence.txt"
dataPath_notebook_label = r"F:\噪声数据处理\实验\Datasets\Sentiment Analysis 中文情感分析\\notebook_label.txt"
dataPath_notebook_sentence = r"F:\噪声数据处理\实验\Datasets\Sentiment Analysis 中文情感分析\\notebook_sentence.txt"
dataPath_phone_label = r"F:\噪声数据处理\实验\Datasets\Sentiment Analysis 中文情感分析\\phone_label.txt"
dataPath_phone_sentence = r"F:\噪声数据处理\实验\Datasets\Sentiment Analysis 中文情感分析\\phone_sentence.txt"


savePath = r"F:\噪声数据处理\实验\Datasets\0-数据集预处理\CommoditySentimentAnalysis-pre"


# 合并camera_label.txt和camera_sentence.txt
camera_label = pd.read_csv(dataPath_camera_label, sep='\t', header=None,dtype=int)
camera_sentence = pd.read_csv(dataPath_camera_sentence, sep='\t', header=None)
camera = pd.concat([camera_label, camera_sentence], axis=1)

# 合并car_label.txt和car_sentence.txt
car_label = pd.read_csv(dataPath_car_label, sep='\t', header=None)
car_sentence = pd.read_csv(dataPath_car_sentence, sep='\t', header=None)
car = pd.concat([car_label, car_sentence], axis=1)

# 合并notebook_label.txt和notebook_sentence.txt
notebook_label = pd.read_csv(dataPath_notebook_label, sep='\t', header=None)
notebook_sentence = pd.read_csv(dataPath_notebook_sentence, sep='\t', header=None)
notebook = pd.concat([notebook_label, notebook_sentence], axis=1)

# 合并phone_label.txt和phone_sentence.txt
phone_label = pd.read_csv(dataPath_phone_label, sep='\t', header=None)
phone_sentence = pd.read_csv(dataPath_phone_sentence, sep='\t', header=None)
phone = pd.concat([phone_label, phone_sentence], axis=1)

# 合并所有数据
data = np.array(pd.concat([camera, car, notebook, phone], axis=0))
# 删除掉data中label为nan的样本
# for low in data:
#     print(low[0])
#     if low[0] != 1.0 and low[0] != 0.0:
#         data = np.delete(data, np.where(data == low)[0], axis=0)

data = np.array([[str(row[0])[0],row[1]] for row in data if row[0]==1.0 or row[0]==0.0])


# 提取data数据中的reviews和labels
reviews = data[:, 1].tolist()   
labels = data[:, 0].tolist()
# 将labels中的值改为int类型

# labels = [int(label) for label in labels]

# 出去review中的所有'\t'字符
for i in range(len(reviews)):
    reviews[i] = reviews[i].replace('\t', ' ')
    
meta_data = []
# 统计各种类标签数量
all_labels_count=tools.GetLabelCount(labels)
meta_data.append('整体类别分布：')
meta_data.append(all_labels_count)

# 将Review进行分词
reviews_token = tools.GetToken(reviews)
# 统计文本长度分布
lengths,lenths_msg=tools.GetTokenLengthDistribution(reviews_token)
#保存lenths_msg
with open(os.path.join(savePath, "lenths_msg.txt"), 'w', encoding='utf-8') as f:
    for line in lenths_msg:
        s=''
        for e in line:
            s+=str(e)+'\t'
        # print(line)
        f.write(s+ '\n')
    f.close()

# 拼接文本列和标签列
reviews = np.array(reviews_token)
labels = np.array(labels)
lengths = np.array(lengths)

CommoditySentimentAnalysis_pre = np.concatenate((reviews.reshape(-1, 1), labels.reshape(-1, 1), lengths.reshape(-1, 1)), axis=1)
# 删除掉CommoditySentimentAnalysis_pre中label为nan的样本
# CommoditySentimentAnalysis_pre=np.array([[row[0],row[1][0],row[2]] for row in CommoditySentimentAnalysis_pre if row[1]=='1.0' or row[1]=='0.0'])

# 将数据随机打乱
np.random.shuffle(CommoditySentimentAnalysis_pre)
#按照8:1:1的比列随机划分为训练集、测试集、验证集
CommoditySentimentAnalysis_train = CommoditySentimentAnalysis_pre[:int(len(CommoditySentimentAnalysis_pre)*0.8),0:2]
CommoditySentimentAnalysis_test = CommoditySentimentAnalysis_pre[int(len(CommoditySentimentAnalysis_pre)*0.8):int(len(CommoditySentimentAnalysis_pre)*0.9),0:2]
CommoditySentimentAnalysis_val = CommoditySentimentAnalysis_pre[int(len(CommoditySentimentAnalysis_pre)*0.9):,0:2]

tra_labels_count=tools.GetLabelCount(CommoditySentimentAnalysis_train[:,1])
meta_data.append('训练集类别分布：')
meta_data.append(tra_labels_count)

test_labels_count=tools.GetLabelCount(CommoditySentimentAnalysis_test[:,1])
meta_data.append('测试集类别分布：')
meta_data.append(test_labels_count)

val_labels_count=tools.GetLabelCount(CommoditySentimentAnalysis_val[:,1])
meta_data.append('验证集类别分布：')
meta_data.append(val_labels_count)

import json
with open(os.path.join(savePath, "class-distribution.txt"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(meta_data, ensure_ascii=False))
    f.close()

# 保存数据

# 如果savePath目录下不存在data文件夹，则创建
if not os.path.exists(os.path.join(savePath, "data")):
    os.makedirs(os.path.join(savePath, "data"))
# 保存训练集、测试集、验证集
# 保存预处理后的数据
with open(os.path.join(savePath, "CommoditySentimentAnalysis-pre.txt"), 'w', encoding='utf-8') as f:
    for line in CommoditySentimentAnalysis_pre:
        s=' '.join(line[0])+'\t'+str(line[1])+'\t'+str(line[2])
        f.write(s+ '\n')
    f.close()

# 如果savePath目录下不存在data文件夹，则创建
if not os.path.exists(os.path.join(savePath, "data")):
    os.makedirs(os.path.join(savePath, "data"))
# 保存训练集、测试集、验证集
with open(os.path.join(savePath, "data", "train.txt"), 'w', encoding='utf-8') as f:
    for line in CommoditySentimentAnalysis_train:
        s=' '.join(line[0])+'\t'+str(line[1])
        f.write(s+ '\n')
    f.close()
with open(os.path.join(savePath, "data", "test.txt"), 'w', encoding='utf-8') as f:
    for line in CommoditySentimentAnalysis_test:
        s=' '.join(line[0])+'\t'+str(line[1])
        f.write(s+ '\n')
    f.close()
with open(os.path.join(savePath, "data", "dev.txt"), 'w', encoding='utf-8') as f:
    for line in CommoditySentimentAnalysis_val:
        s=' '.join(line[0])+'\t'+str(line[1])
        f.write(s+ '\n')
    f.close()

if not os.path.exists(os.path.join(savePath, "log")):
    os.makedirs(os.path.join(savePath, "log"))
if not os.path.exists(os.path.join(savePath, "saved_dict")):
    os.makedirs(os.path.join(savePath, "saved_dict"))

print('CommoditySentimentAnalysis Preprocess Done!')
