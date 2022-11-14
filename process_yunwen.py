import pandas as pd

# 设置label
label = {0:"韵文",1:"散文"}

# 导入数据
path = './data/Bert-Chinese-Text-Classification-Pytorch-master/Poetry/宋_1.csv'
df = pd.read_csv(path)

# 散文数据:1048576条，抽取30000条
# 韵文数据：训练集:唐10000条，宋10000条，近现代10000条、测试集：3000条、验证集：3000条


# 数据集随机取样
df = df.sample(frac=1.0).reset_index(drop=True)
df = df.iloc[:,3:]
df["label"] = 0
# print(df)

train_df = df.iloc[0:10000:,]
val_df = df.iloc[10001:13000,]
test_df = df.iloc[13001:16000,]

train_df.to_csv('./data/Bert-Chinese-Text-Classification-Pytorch-master/xian_train.csv')
val_df.to_csv('./data/Bert-Chinese-Text-Classification-Pytorch-master/xian_val.csv')
test_df.to_csv('./data/Bert-Chinese-Text-Classification-Pytorch-master/xian_test.csv')
