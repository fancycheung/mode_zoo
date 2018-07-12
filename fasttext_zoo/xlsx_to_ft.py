# -*- coding: utf-8 -*-
import pandas as pd
import argparse
import os
from procress_data import clear_en_line,clear_zh_line,punctuation_dict,clean_zh_seg_coment
import jieba
import numpy as np


repls = punctuation_dict()

def xlsx_to_ft_format_label(file,f_train,f_eval,lang):
    '''从标注数据excel表格转为fasttext格式，当前是用16个一级标签，以后训练样本多了可尝试200个的三级标签'''
    
    if lang == "zh":
        print("load zh!")
        data = pd.read_excel(file,sheet_name="中文1",header=0)
    elif lang == "en":
        print("load en!")
        data = pd.read_excel(file,sheet_name="英文1",header=0)
    data.columns = ["id","comment","label","sub_label"]
    label_set = ['卫生一般','服务周到','房间设施好','餐饮满意','位置优越','服务水平低','卫浴满意','酒店设施一般','卫浴不满意','餐饮不满意','房间设施一般','交通环境一般','酒店设施好','卫生干净','交通便利']
    cache_list = []
    for i in range(len(data)):
        comment = data.iloc[i,1]
        if lang == 'zh':
            comment = clear_zh_line(comment)
        elif lang== 'en':
            comment = clear_en_line(comment)

        label = data.iloc[i,2]
        if str(label)== "nan" or str(label)== "null" or str(label).strip()=="NULL":
            label = "NULL"
            final_label = "__label__"+label
        else:
            split_label = label.split("|")
            split_label = [x.strip() for x in split_label if len(x)>2]
            for j in split_label:
               # print(i,j)
                assert j in label_set,"line %d" % i
            final_label = " ".join(["__label__"+x for x in split_label])

        if lang== "zh" and i > 820 and i <1000: # 当前820到1000的样本标注有问题，过滤掉。
                continue
        if lang == "zh":
            seg_comment = jieba.cut(comment)
            comment = clean_zh_seg_coment(seg_comment)
        line = final_label + " " + comment.lower().strip()
        cache_list.append(line)
    split = int(np.floor(len(cache_list)*(1-0.11)))
    with open(f_train,'w') as ft:
        for line in cache_list[:split]:
            ft.write(line+"\n")
    with open(f_eval,'w') as fe:
        for line in cache_list[split:]:
            fe.write(line+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",default="")
    parser.add_argument("--output",default="./data/")
    parser.add_argument("--lang",default="en")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    xlsx_to_ft_format_label(args.input,args.output+"ft_%s.train" % args.lang,args.output+"ft_%s.eval" % args.lang,args.lang)






