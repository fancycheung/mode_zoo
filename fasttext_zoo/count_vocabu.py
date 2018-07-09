#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:01:06 2018

@author: miaoji
"""
import os
import pickle

def count_vocabu(file):
    cache_path = "/data/caozhaojun/true_procress_data/procressed_data/cache_vocabu_count.pickle"
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        with open(cache_path, 'rb') as data_f:
            vocabu_count = pickle.load(data_f)
        return vocabu_count
    
    else:
        with open(file,'r') as f:
            vocabu_count = {}
            for line in f.readlines():
                line = line.split()
                for word in line:
                    if vocabu_count.get(word,None) is None:
                        vocabu_count[word] = 1
                    else:
                        vocabu_count[word] += 1

        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'ab') as data_f:
                pickle.dump(vocabu_count,data_f)
                
    return vocabu_count

def sub_word_vec_from_wiki(in_file,out_file):

    with open(out_file,'w') as out_f:
        with open(in_file,'r') as f:
            for line in f.readlines():
                word,_ = line.split(' ',1)
                if word in vocabu_count:
                    out_f.write(line)


def count_label_class(file,save=0,out_file = None):
    '''count label class '''
    with open(file,'r') as f:
        label_count_dict = {}
        for line in f.readlines():
            label = line.split(' ',1)[0].split('__label__')[1]
            if label not in label_count_dict:
                label_count_dict[label] = 1
            else:
                label_count_dict[label] += 1
    if save == 1:
        with open(out_file,'w') as out_f:
            for k,v in label_count_dict.items():
                out_f.write(k+':'+str(v)+'\n')

    return label_count_dict


if __name__ == "__main__":
    '''
    file = '/data/zhangbin/caozhaojun/true_procress_data/procressed_data/procress_en_data_final.txt'
    vocabu_count = count_vocabu(file)
    print('length',len(vocabu_count))
    print('hotel',vocabu_count['good'])

    in_file = '/data/zhangbin/caozhaojun/word2vec/wiki-news-300d-1M-subword.vec'    
    out_file = '/data/zhangbin/caozhaojun/true_procress_data/procressed_data/sub_word_vec_from_wiki.vec'
    sub_word_vec_from_wiki(in_file,out_file)
    '''
    file = './procressed_data/ft_format_data_tmp.train'
    out_file = './procressed_data/label_class_count.txt'
    label_count_dict = count_label_class(file,1,out_file)
    print('count_label_class_len',len(label_count_dict))
