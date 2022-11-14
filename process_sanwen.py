import numpy as np
import pandas as pd
import re
import csv

#读取文件
txt = open("./data/Bert-Chinese-Text-Classification-Pytorch-master/shibu_clean/shibu_clean.txt", 'r', encoding='utf-8')

trainfile=open("./data/Bert-Chinese-Text-Classification-Pytorch-master/SanWen/sanwen_train.csv", 'w', newline='', encoding='utf-8')
valfile=open("./data/Bert-Chinese-Text-Classification-Pytorch-master/SanWen/sanwen_val.csv", 'w', newline='', encoding='utf-8')
testfile=open("./data/Bert-Chinese-Text-Classification-Pytorch-master/SanWen/sanwen_test.csv", 'w', newline='', encoding='utf-8')

trainwriter = csv.writer(trainfile)
valwriter = csv.writer(valfile)
testwriter = csv.writer(testfile)

trainlist = []
vallist = []
testlist = []

header = ['content', 'label']

trainwriter.writerow(header)
valwriter.writerow(header)
testwriter.writerow(header)

for i in range(10000):
    passage = txt.readline().strip()
    # 特殊符号不应成为区分两者的标志，所以应当去掉特殊符号，和韵文保持一致，只留下逗号和句号
    lines = re.sub(r'[《》（）〈〉〔〕“”\u3000﹑‘’【】0-9]+','', passage)
    trainlist.append([lines,1])

for i in range(3000):
    passage = txt.readline().strip()
    # 特殊符号不应成为区分两者的标志，所以应当去掉特殊符号，和韵文保持一致，只留下逗号和句号
    lines = re.sub(r'[《》（）〈〉〔〕“”\u3000﹑‘’【】0-9]+','', passage)
    vallist.append([lines,1])

for i in range(3000):
    passage = txt.readline().strip()
    # 特殊符号不应成为区分两者的标志，所以应当去掉特殊符号，和韵文保持一致，只留下逗号和句号
    lines = re.sub(r'[《》（）〈〉〔〕“”\u3000﹑‘’【】0-9]+','', passage)
    testlist.append([lines,1])

trainwriter.writerows(trainlist)
valwriter.writerows(vallist)
testwriter.writerows(testlist)