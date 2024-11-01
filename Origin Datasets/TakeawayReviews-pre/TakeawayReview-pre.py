import importlib
import os
import numpy as np
import pandas as pd
import sys

dataPath = r"F:\噪声数据处理\实验\Datasets\外卖评价数据\waimai_10k.csv"

savePath = r"F:\噪声数据处理\实验\Datasets\0-数据集预处理\TakeawayReviews-pre"

sys.path.append("../")
tools = importlib.import_module("data_preprocessing_tools")

meta_data = []
# 读取数据
data = pd.read_csv(dataPath, encoding='utf-8')

# 获取文本列和标签列
reviews = data['review'].tolist()
labels = data['label'].tolist()

# 出去review中的所有'\t'字符
for i in range(len(reviews)):
    reviews[i] = reviews[i].replace('\t', ' ')

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


# 拼接文本列/标签列/文本长度
reviews = np.array(reviews_token)
labels = np.array(labels)
lengths = np.array(lengths)
TakeawayReview_pre = np.concatenate(
    (reviews.reshape(-1, 1), labels.reshape(-1, 1), lengths.reshape(-1, 1)), axis=1
)


# 将数据随机打乱
np.random.shuffle(TakeawayReview_pre)
# 按照8:1:1的比列随机划分为训练集、测试集、验证集
TakeawayReview_train = TakeawayReview_pre[:int(
    len(TakeawayReview_pre)*0.8),0:2]
TakeawayReview_test = TakeawayReview_pre[int(
    len(TakeawayReview_pre)*0.8):int(len(TakeawayReview_pre)*0.9),0:2]
TakeawayReview_val = TakeawayReview_pre[int(
    len(TakeawayReview_pre)*0.9):,0:2]

tra_labels_count=tools.GetLabelCount(TakeawayReview_train[:,1])
meta_data.append('训练集类别分布：')
meta_data.append(tra_labels_count)

test_labels_count=tools.GetLabelCount(TakeawayReview_test[:,1])
meta_data.append('测试集类别分布：')
meta_data.append(test_labels_count)

val_labels_count=tools.GetLabelCount(TakeawayReview_val[:,1])
meta_data.append('验证集类别分布：')
meta_data.append(val_labels_count)

#长度分布
# meta_data.append('训练集长度分布：')
# meta_data.append(lenths_msg)

import json
with open(os.path.join(savePath, "meta_data.txt"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(meta_data, ensure_ascii=False))
    f.close()

# 保存预处理后的数据
with open(os.path.join(savePath, "TakeawayReview-pre.txt"), 'w', encoding='utf-8') as f:
    for line in TakeawayReview_pre:
        s=' '.join(line[0])+'\t'+str(line[1])+'\t'+str(line[2])
        f.write(s+ '\n')
    f.close()

# 如果savePath目录下不存在data文件夹，则创建
if not os.path.exists(os.path.join(savePath, "data")):
    os.makedirs(os.path.join(savePath, "data"))
# 保存训练集、测试集、验证集
with open(os.path.join(savePath, "data", "train.txt"), 'w', encoding='utf-8') as f:
    for line in TakeawayReview_train:
        s=' '.join(line[0])+'\t'+str(line[1])
        f.write(s+ '\n')
    f.close()
with open(os.path.join(savePath, "data", "test.txt"), 'w', encoding='utf-8') as f:
    for line in TakeawayReview_test:
        s=' '.join(line[0])+'\t'+str(line[1])
        f.write(s+ '\n')
    f.close()
with open(os.path.join(savePath, "data", "dev.txt"), 'w', encoding='utf-8') as f:
    for line in TakeawayReview_val:
        s=' '.join(line[0])+'\t'+str(line[1])
        f.write(s+ '\n')
    f.close()

if not os.path.exists(os.path.join(savePath, "log")):
    os.makedirs(os.path.join(savePath, "log"))
if not os.path.exists(os.path.join(savePath, "saved_dict")):
    os.makedirs(os.path.join(savePath, "saved_dict"))

print('Takeaway Reviews Preprocess Done!')
