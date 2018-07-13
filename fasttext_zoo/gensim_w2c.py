#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:48:54 2018

@author: miaoji
"""
import os
import gensim
from gensim.models import word2vec

if __name__ == "__main__":
    sent = word2vec.LineSentence("/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_zh_nolabel.txt")#需要自己\
           # 制作训练集
    model = word2vec.Word2Vec(sent,size=200,window=5,min_count=5,workers=6,iter=10)
#    model.save('daodao_zh_word_vec_model')
#    model = word2vec.Word2Vec.load('daodao_zh_word_vec.txt')
    model.wv.save_word2vec_format('daodao_zh_word2vec.txt', binary=False) #保存训练好的词向量
    
