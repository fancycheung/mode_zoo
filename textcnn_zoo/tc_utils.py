# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:42:14 2018

@author: miaoji
"""
#import fasttext
import langid
import codecs
import numpy as np
import os
import pickle
from tqdm import tqdm
import re
import argparse

def batch_iter(x, y, batch_size=500):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size)

    indices = np.random.permutation(np.arange(data_len))
#    x_shuffle = x[indices]
#    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
#        end_id = min((i + 1) * batch_size, data_len)
        end_id = (i + 1) * batch_size
        
        yield x[indices[start_id:end_id]], y[indices[start_id:end_id]]

def softmax(logit):
    probs = []
    for x in logit:
        soft_x = np.exp(x)/np.sum(np.exp(x),axis=0)
        probs.append(list(soft_x))

    return probs

def evaluate(predict_labels,true_labels):
    '''eivaluate test result'''
    eval_counter,eval_acc,eval_p,eval_r,eval_f1 = 0,0.0,0.0,0.0,0.0

    for i in range(len(true_labels)):
        true_y = true_labels[i]
        predict_y = predict_labels[i]
        y = set(np.where(true_y==1)[0])
#        y_ = set(np.where(predict_y==1))
        if not isinstance(predict_y,list):
            predict_y = [predict_y]
        y_ = set(predict_y)
        acc = len(y & y_) / (len(y|y_))
        precision = len(y & y_) / (len(y_))
        recall = len(y & y_) / (len(y))
        f1 = 2.0 * precision * recall / (precision + recall + 0.000001)
    #    if counter % 3 == 0:
    #        print(counter,":",y,y_)
        eval_counter,eval_acc,eval_p,eval_r,eval_f1 = eval_counter+1,eval_acc+acc,eval_p+precision,eval_r+recall,eval_f1+f1
        
    return eval_acc/float(eval_counter),eval_p/float(eval_counter),eval_r/float(eval_counter),eval_f1/float(eval_counter)



def load_word2vec(file,cache_path = "",n=600000):
    '''加载、缓存词向量,选择常用部分'''
    print("start loading word2vec ")
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path): #如果缓存文件存在，则直接读取
        with open(cache_path, 'rb') as data_f:
            word2vec_dict = pickle.load(data_f)
            print('word2vec_dict_length',len(word2vec_dict))
            return word2vec_dict
    
    else:
        word2vec_dict = {}
        open_f = codecs.open(file, 'r', 'utf8')
        line_id = 0
        for line in open_f:
            line_id += 1
            if line_id > 0 or line_id < n:
                word,vec = line.strip().split(' ',1)
                vec = [np.float(v) for v in vec.split()]                
                if line_id < 3:
                    print(type(vec),vec)
                word2vec_dict[word] = vec
            if line_id == n:
                print(line_id,'break')
                break
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'ab') as data_f:
                pickle.dump(word2vec_dict,data_f)
        print('word2vec_dict_length',len(word2vec_dict))

    return word2vec_dict


def create_voabulary(file,cache_path = "",from_word2vec=1):
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path): #如果缓存文件存在，则直接读取
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word=pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}

        vocabulary_word2index['PAD_ID']=0
        vocabulary_index2word[0]='PAD_ID'

        open_f = codecs.open(file,'r','utf-8')
        lines = open_f.readlines()

        if from_word2vec == 1:
            for i,line in enumerate(lines):
                if i > 0:
                    vocab,_ = line.split(' ',1)
                    vocabulary_word2index[vocab] = i
                    vocabulary_index2word[i] = vocab

        else:
            vocabu_count = {}
            pattern = re.compile(r"(\b__label__(\S+))")
            for i,line in enumerate(lines):  #count word dict
                find = re.findall(pattern,line)
                label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                words = comment.split()
                for word in words:
                    if vocabu_count.get(word,None) is None:
                        vocabu_count[word] = 1
                    else:
                        vocabu_count[word] += 1
            sorted_vocab = sorted(vocabu_count.items(),key=lambda x:x[1],reverse=True)
            print('sorted_vocab',len(sorted_vocab))
            for i,item in enumerate(sorted_vocab):
                if i < 20:
                    print(item,item[0])
                vocabulary_word2index[item[0]] = i+1
                vocabulary_index2word[i+1] = item[0]
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word), data_f)

    return vocabulary_word2index,vocabulary_index2word



def create_voabulary_label():
    file = "/data/caozhaojun/true_procress_data/label.mode.txt"
    cache_path = "./cache_pickle/cache_word_voabulary_label.pickle"
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label=pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label
    else:
        vocabulary_word2index_label={}
        vocabulary_index2word_label={}
        open_f = codecs.open(file, 'r', 'utf8')
        lines = open_f.readlines()
        label_id = 0
        for line in lines:
            if line.strip() == '' or line[0] == '#':
                continue
            else:
                infos = line.strip().split('\t')
                label = infos[1]
                vocabulary_word2index_label[label] = label_id
                vocabulary_index2word_label[label_id] = label
                label_id += 1
        vocabulary_word2index_label['NULL'] = label_id        
        vocabulary_index2word_label[label_id] = 'NULL'

        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index_label,vocabulary_index2word_label), data_f)
    return vocabulary_word2index_label,vocabulary_index2word_label

def create_voab_label():
    "新指标"
    label_set = ['卫生一般','服务周到','房间设施好','餐饮满意','位置优越','服务水平低','卫浴满意','酒店设施一般','卫浴不满意','餐饮不满意','房间设施一般','交通环境一般','酒店设施好','卫生干净','交通便利','NULL']
    vocabulary_word2index_label={}
    vocabulary_index2word_label={}
    for i,label in enumerate(label_set):
        vocabulary_word2index_label[label] = i
        vocabulary_index2word_label[i] = label
    return vocabulary_word2index_label,vocabulary_index2word_label


def load_data(vocabulary_word2index,vocabulary_word2index_label,training_data_path,multi_label_flag=False,cache_path = ""):
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        with open(cache_path, 'rb') as data_f:
            X, Y = pickle.load(data_f)
            return X, Y
    else:
        open_f = codecs.open(training_data_path, 'r', 'utf8')
        lines = open_f.readlines()
        X = []
        Y = []
        pattern = re.compile(r"(\b__label__(\S+))")
        for i, line in enumerate(lines):
            find = re.findall(pattern,line)
            label = [item[1] for item in find]
            label_len = ' '.join([item[0] for item in find])
            comment = line[len(label_len):].strip()
            x = comment
            y = label
            x = x.replace('\n','')
            if i < 5:
                print('-------------------------')
                print("x0:",x)
                print('y0:',y)
            x = x.split(" ")
            x = [vocabulary_word2index.get(e,0) for e in x]
            label_y = []
            for it_y in y:
                it_y = vocabulary_word2index_label[it_y]
                label_y.append(it_y)

            if i < 5:
                print("x1:",x) #word to index
                print('label_y:',label_y)
            if len(label_y) < 1:
                print("##########################",i)
            X.append(x)
            Y.append(label_y)

        number_examples = len(X)
        print("number_examples:",number_examples) 

        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((X,Y), data_f)
    return X,Y


def load_data_predict(vocabulary_word2index,vocabulary_word2index_label,predict_data_path):
    open_f = codecs.open(predict_data_path, 'r', 'utf8')
    lines = open_f.readlines()
    X = []
    for i, line in enumerate(lines):
        line = line.replace('\n','')
        if i<5:
            print('-------------------------')
            print("x0:",line) 
        line = line.split(" ")
        line = [vocabulary_word2index.get(e,0) for e in line]
        if i<5:
            print("x1:",line) #word to index
        X.append(line)
        
    number_examples = len(X)
    print("predict_examples:",number_examples) 
    return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang',default='en')
    args = parser.parse_args()
    lang = args.lang
    if lang == 'en':
        file = "./cache_pickle/multi_ft_en_train_cbow_300d.vec"
    elif lang == 'zh':
        file = "./cache_pickle//daodao_zh_word2vec.txt"
    word2vec_dict = load_word2vec(file,cache_path = "./cache_pickle/ft_train_word2vec_%s.pickle" % lang)
    print('word2vec_dict_length',len(word2vec_dict))
   
    vocabulary_word2index_label,_ = create_voab_label()
    print('vocabulary_word2index_label',len(vocabulary_word2index_label))
    print(vocabulary_word2index_label)

#    file = "/data/caozhaojun/fasttext_model/multi_ft_en_train_cbow_300d.vec"
    vocabulary_word2index, vocabulary_index2word = create_voabulary(file,cache_path = "./cache_pickle/ft_%s_voabulary.pickle" % lang,from_word2vec=1)
    vocab_size = len(vocabulary_word2index)
    print("vocab_size",vocab_size)
    for i in range(5):
        print("vocabulary_index2word[%d]" % i ,vocabulary_index2word[i])
#    print("vocabulary_word2index['hotel']",vocabulary_word2index['hotel'])
    
    file = "./cache_pickle/ft_%s.train" % lang
    load_data(vocabulary_word2index, vocabulary_word2index_label,file,cache_path = "./cache_pickle/train_data_%s.pickle" % lang)
    file = "./cache_pickle/ft_%s.eval" % lang
    load_data(vocabulary_word2index, vocabulary_word2index_label,file,cache_path = "./cache_pickle/eval_data_%s.pickle" % lang)



