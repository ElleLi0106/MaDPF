import os
import numpy as np
import importlib
import pandas as pd
import sys

sys.path.append("../")
tools = importlib.import_module("data_preprocessing_tools")

dataPath_test_neg = r"F:\噪声数据处理\实验\Datasets\IMDb Movie Reviews\aclImdb_v1\test\neg"
dataPath_test_pos = r"F:\噪声数据处理\实验\Datasets\IMDb Movie Reviews\aclImdb_v1\test\pos"

dataPath_train_neg = r"F:\噪声数据处理\实验\Datasets\IMDb Movie Reviews\aclImdb_v1\train\neg"
dataPath_train_pos = r"F:\噪声数据处理\实验\Datasets\IMDb Movie Reviews\aclImdb_v1\train\pos"

savePath = r"F:\噪声数据处理\实验\Datasets\0-数据集预处理\IMDbMoviesReviews-pre"


#读取训练集和测试集的负样本和正样本
train_neg_files = os.listdir(dataPath_train_neg)
train_pos_files = os.listdir(dataPath_train_pos)
test_neg_files = os.listdir(dataPath_test_neg)
test_pos_files = os.listdir(dataPath_test_pos)

# 读取训练集和测试集的负样本和正样本
train_neg_reviews = [open(os.path.join(dataPath_train_neg, file), 'r', encoding='utf-8').read() for file in train_neg_files]
train_pos_reviews = [open(os.path.join(dataPath_train_pos, file), 'r', encoding='utf-8').read() for file in train_pos_files]
test_neg_reviews = [open(os.path.join(dataPath_test_neg, file), 'r', encoding='utf-8').read() for file in test_neg_files]
test_pos_reviews = [open(os.path.join(dataPath_test_pos, file), 'r', encoding='utf-8').read() for file in test_pos_files]

# 将训练集和测试集的负样本和正样本合并
reviews = train_neg_reviews + train_pos_reviews + test_neg_reviews + test_pos_reviews
labels = np.array([0]*len(train_neg_reviews) + [1]*len(train_pos_reviews) + [0]*len(test_neg_reviews) + [1]*len(test_pos_reviews))

# 出去review中的所有'\t'字符
for i in range(len(reviews)):
    reviews[i] = reviews[i].replace('\t', ' ')
    
meta_data = []
# 统计各种类标签数量
all_labels_count=tools.GetLabelCount(labels)
meta_data.append('整体类别分布：')
meta_data.append(all_labels_count)


# 统计文本长度分布
lengths=[]
lenths_msg=[]
for i in range(len(reviews)):
    # 以空格分割文本
    lengths.append(len(reviews[i].split(' ')))

print("平均长度：", np.mean(lengths))
lenths_msg.append(["平均长度：",np.mean(lengths)])
print("最大长度：", np.max(lengths))
lenths_msg.append(["最大长度：",np.max(lengths)])
print("最小长度：", np.min(lengths))
lenths_msg.append(["最小长度：",np.min(lengths)])

#"长度分布："
values,keys=np.histogram(lengths, bins=max(lengths)) # 统计每个长度的数量
lenths_msg.append(["长度分布："])
for key,value in zip(keys,values):
    # print(key,value)
    # print(int(key),int(value))
    #对key进行四舍五入不保留小数
    key = round(key)
    lenths_msg.append([int(key),int(value)])

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
reviews = np.array(reviews)
labels = np.array(labels)
lengths = np.array(lengths)

IMDbMoviesReviews_pre = np.concatenate((reviews.reshape(-1, 1), labels.reshape(-1, 1), lengths.reshape(-1, 1)), axis=1)


#保存lenths_msg
with open(os.path.join(savePath, "lenths_msg.txt"), 'w', encoding='utf-8') as f:
    for line in lenths_msg:
        s=''
        for e in line:
            s+=str(e)+'\t'
        # print(line)
        f.write(s+ '\n')
    f.close()


# 将数据随机打乱
np.random.shuffle(IMDbMoviesReviews_pre)
#按照8:1:1的比列随机划分为训练集、测试集、验证集
IMDbMoviesReviews_train = IMDbMoviesReviews_pre[:int(len(IMDbMoviesReviews_pre)*0.8),0:2]
IMDbMoviesReviews_test = IMDbMoviesReviews_pre[int(len(IMDbMoviesReviews_pre)*0.8):int(len(IMDbMoviesReviews_pre)*0.9),0:2]
IMDbMoviesReviews_val = IMDbMoviesReviews_pre[int(len(IMDbMoviesReviews_pre)*0.9):,0:2]

tra_labels_count=tools.GetLabelCount(IMDbMoviesReviews_train[:,1])
meta_data.append('训练集类别分布：')
meta_data.append(tra_labels_count)

test_labels_count=tools.GetLabelCount(IMDbMoviesReviews_test[:,1])
meta_data.append('测试集类别分布：')
meta_data.append(test_labels_count)

val_labels_count=tools.GetLabelCount(IMDbMoviesReviews_val[:,1])
meta_data.append('验证集类别分布：')
meta_data.append(val_labels_count)

import json
with open(os.path.join(savePath, "meta_data.txt"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(meta_data, ensure_ascii=False))
    f.close()

# 保存预处理后的数据
# with open(os.path.join(savePath, "IMDbMoviesReviews-pre.txt"), 'w', encoding='utf-8') as f:
#     for line in IMDbMoviesReviews_pre:
#         s=' '.join(line[0])+'\t'+str(line[1])+'\t'+str(line[2])
#         f.write(s+ '\n')
#     f.close()
np.savetxt(os.path.join(savePath, "IMDbMoviesReviews-pre.txt"), IMDbMoviesReviews_pre, fmt='%s',delimiter='\t',encoding='utf-8')

# 如果savePath目录下不存在data文件夹，则创建
if not os.path.exists(os.path.join(savePath, "data")):
    os.makedirs(os.path.join(savePath, "data"))
# 保存训练集、测试集、验证集
# with open(os.path.join(savePath, "data", "train.txt"), 'w', encoding='utf-8') as f:
#     for line in IMDbMoviesReviews_train:
#         s=' '.join(line[0])+'\t'+str(line[1])
#         f.write(s+ '\n')
#     f.close()
# with open(os.path.join(savePath, "data", "test.txt"), 'w', encoding='utf-8') as f:
#     for line in IMDbMoviesReviews_test:
#         s=' '.join(line[0])+'\t'+str(line[1])
#         f.write(s+ '\n')
#     f.close()
# with open(os.path.join(savePath, "data", "dev.txt"), 'w', encoding='utf-8') as f:
#     for line in IMDbMoviesReviews_val:
#         s=' '.join(line[0])+'\t'+str(line[1])
#         f.write(s+ '\n')
#     f.close()
np.savetxt(os.path.join(savePath, "data", "train.txt"), IMDbMoviesReviews_train, fmt='%s',delimiter='\t',encoding='utf-8')
np.savetxt(os.path.join(savePath, "data", "test.txt"), IMDbMoviesReviews_test, fmt='%s',delimiter='\t',encoding='utf-8')
np.savetxt(os.path.join(savePath, "data", "dev.txt"), IMDbMoviesReviews_val, fmt='%s',delimiter='\t',encoding='utf-8')


if not os.path.exists(os.path.join(savePath, "log")):
    os.makedirs(os.path.join(savePath, "log"))
if not os.path.exists(os.path.join(savePath, "saved_dict")):
    os.makedirs(os.path.join(savePath, "saved_dict"))

print('IMDb Movies Reviews Preprocess Done!')
