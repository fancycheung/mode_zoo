#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:25:19 2018

@author: miaoji
"""
import nltk.tokenize as nt
from textblob import TextBlob
import time
import os
import jieba 
import nltk
from tqdm import tqdm 
from snownlp import normal
from snownlp import seg
from snownlp.summary import textrank
import re
from random import shuffle


def clean_str(string):
    '''英文文本终极处理'''
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " ( ", string) 
    string = re.sub(r"\)", " ) ", string) 
    string = re.sub(r"\?", " ? ", string) 
    string = re.sub(r"\s{2,}", " ", string) 
    #add
    string = re.sub(r"\.{2,}", ".", string)
    string = re.sub(r"!{2,}", "!", string)
    string = re.sub(r"-{2,}", "-", string)
    string = re.sub(r"\-", " - ", string)
    return string.strip().lower()

def procress_en_data(file,out_file):
    '''处理英文文件'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            all_line = in_f.readlines()
            for line in tqdm(all_line):
                new_line = clean_str(line)
                out_f.write(new_line+'\n')

def procress_zh_data(file,out_file):
    '''处理中文文件'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            for line in tqdm(in_f.readlines()):
                text = normal.zh2hans(line)
                text = normalize_text(text)
                text = clean_zh(text)
                out_f.write(text)

def comment_sents_en_id(file,out_file):
    '''英文文本初级处理，将一个评论按句分割，返回：行id + 分句id + 评论'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            line_id = 0
            all_line = in_f.readlines()
            tmp_len = len(all_line) - 999
            for line in tqdm(all_line):
                line_id += 1
                line = line.replace("\n","")
                text = line.replace(". ","\n")
                text = re.sub(r"\n{2,}","\\n",text)
                sents = text.split("\n")
                sent_id = 1
                for item in sents:
                    if len(item) < 3: #过滤长度小于3的行
                        continue
                    prefix = str(line_id)+'-'+str(sent_id)
                    sent_id += 1
                    final_line = prefix+' '+item+"\n"
                    final_line = re.sub(r"\n{2,}","\\n",final_line)
                    out_f.write(final_line.lower())

def clear_en_line(line):
    '''处理英文，过滤换行符'''
    line = clean_str(line)
    line = line.replace("\n","")
    return line

def comment_sents_en_line(line):
    '''英文测试集处理，输入一个评论，输出处理后且分句的list'''
    line_list = []
    line = clear_en_line(line)
    text = line.replace(". ","\n")
    if text[-1] == '.':
        text =  text[:-1]
    text = re.sub(r"\n{2,}","\\n",text)
    sents = text.split("\n")
    for item in sents:
        final_line = item+"\n"
        final_line = re.sub(r"\n{2,}","\\n",final_line)
        line_list.append(final_line.lower())
    return line_list

def clear_zh_line(line):
    '''处理中文文本'''
    text = normal.zh2hans(line)
    text = normalize_text(text)
    text = clean_zh(text)
    text = text.replace("\n","")
    return text

def comment_sents_zh_line(line):
    '''中文测试集处理，输入一个评论，输出处理后且分句的list'''
    line_list = []
    text = clear_zh_line(line)
    text = text.replace("。","\n")
    text = re.sub(r"\n{2,}","\\n",text)
    sents = text.split("\n")
    for item in sents:
        final_line = item+"\n"
        final_line = re.sub(r"\n{2,}","\\n",final_line)
        line_list.append(final_line.lower())
    return line_list


def comment_sents_zh_id(file,out_file,flag=True):
    '''中文文本初级处理，将一个评论按句分割，返回：行id + 分句id + 评论'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            line_id = 0
            all_line = in_f.readlines()
            tmp_len = len(all_line) - 499
            for line in tqdm(all_line):
                line_id += 1
                if flag==True:
                    text = normal.zh2hans(line)
                    text = normalize_text(text)
                    text = clean_zh(text)
                    text = text.replace("\n","")
                    text = text.replace("。","\n")
                    text = re.sub(r"\n{2,}","\\n",text)
                    sents = text.split("\n")
                    sent_id = 1
                    for item in sents:
                        if len(item) < 3: #过滤长度小于3的行
                            continue
                        else:
                            prefix = str(line_id)+'-'+str(sent_id)
                            sent_id += 1
                            final_line = prefix+' '+item+"\n"
                            final_line = re.sub(r"\n{2,}","\\n",final_line)
                            out_f.write(final_line.lower())


def extract_test_data(file,out_file,n=None):
    '''从初级处理后的文件中，选取后面行，去掉行id，制作测试集'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            for line in tqdm(in_f.readlines()):
                line_id = line.split(' ',1)[0]
                line_id = line_id.split('-')[0]
                if int(line_id) > n:
                    out_f.write(line)

def punctuation_dict():
  '''中文特殊符号'''
  cpun = [['	'],
          [u'﹗'],
          [u'“', u'゛', u'〃', u'′'],
          [u'”'],
          [u'´', u'‘', u'’'],
          [u'；', u'﹔'],
          [u'《', u'〈', u'＜'],
          [u'》', u'〉', u'＞'],
          [u'﹑'],
          [u'【', u'『', u'〔', u'﹝', u'｢', u'﹁'],
          [u'】', u'』', u'〕', u'﹞', u'｣', u'﹂'],
          [u'（', u'「'],
          [u'）', u'」'],
          [u'﹖'],
          [u'︰', u'﹕'],
          [u'・', u'．', u'·', u'‧', u'°'],
          [u'●', u'○', u'▲', u'◎', u'◇', u'■', u'□', u'※', u'◆'],
          [u'〜', u'～', u'∼'],
          [u'︱', u'│', u'┼'],
          [u'╱'],
          [u'╲'],
          [u'—', u'ー', u'―', u'‐', u'−', u'─', u'﹣', u'–', u'ㄧ']]
  epun = [u' ', u'！', u'"', u'"', u'\'', u';', u'<', u'>', u'、', u'[', u']', u'(', u')', u'？', u'：', u'･', u'•', u'~', u'|', u'/', u'\\', u'-']
  repls = {}

  for i in range(len(cpun)):
    for j in range(len(cpun[i])):
      repls[cpun[i][j]] = epun[i]

  return repls

repls = punctuation_dict()

def normalize_text(text):
  '''处理中文，归一化特殊符号，全角转半角'''
  text = replace_all(repls,text)
  text = "".join([Q2B(c) for c in list(text)])
  return text

def replace_all(repls, text):
  # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], text)
  return re.sub(u'|'.join(re.escape(key) for key in repls.keys()),
                lambda k: repls[k.group(0)], text)

def Q2B(char):
  """全角转半角"""
  inside_code = ord(char)
  if inside_code == 0x3000:
    inside_code = 0x0020
  else:
    inside_code -= 0xfee0
  if inside_code < 0x0020 or inside_code > 0x7e:
    return char
  return chr(inside_code)

def clean_zh(text):
    '''去掉重复符号'''
    text = re.sub(r"-{1,}", ",",text)
    text = re.sub(r"!{2,}", "!",text)
    text = re.sub(r"。{2,}", "。",text)
    text = re.sub(r"…{2,}", ",",text)
    text = re.sub(r"･{2,}", "･",text)
    text = re.sub(r"\.{2,}", ".",text)
    return text

def clean_zh_seg_coment(line):
    '''中文终极处理，去掉数字、英文等'''
    sent_comment = " ".join(line)#zh
    sent_comment = sent_comment.replace("!","") #z
    sent_comment = sent_comment.replace("?","") #zh
    sent_comment = sent_comment.replace(",","") #zh
    sent_comment = sent_comment.replace("-","") #zh
    sent_comment = sent_comment.replace("#","") #zh
    sent_comment = sent_comment.replace("(","") #zh
    sent_comment = sent_comment.replace(")","") #zh
    sent_comment = sent_comment.replace(".","") #zh
    sent_comment = sent_comment.replace("、","") #zh
    sent_comment = re.sub(r"[A-Za-z0-9/\'\"]","",sent_comment)
    sent_comment = re.sub(r"\s+"," ",sent_comment).strip() #zh
    return sent_comment

def make_test_data_for_evaluate(infile,outfile,lang="en"):
    '''从fasttext格式数据集中，处理分别处理中英文本，中文需要分词'''
    pattern = re.compile(r"(\b__label__(\S+))")
    with open(outfile,'w') as out_f:
        with open(infile,'r') as in_f:
            for line in in_f.readlines():
                find = re.findall(pattern,line)
                #label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                if lang == "en":
                    comment = clean_str(comment)
                elif lang =="zh":
                    seg_comment = jieba.cut(comment)
                    comment = clean_zh_seg_coment(seg_comment)
                final_label = line[:len(label_len)].strip() 
                out_f.write(final_label+" "+comment+"\n")

def create_null_testdata(file,out_file):
    '''输入是有多标签前缀的文本，输出无标签评论'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as f:
            pattern = re.compile(r"(\b__label__(\S+))")
            for line in f.readlines():
                find = re.findall(pattern,line)
                #label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                comment = comment +'\n'
                out_f.write(comment)

def remove_null(file,out_file,out_null):
    '''对null and nonull分类'''
    with open(out_null,'w') as null_f:
        with open(out_file,'w') as out_f:
            with open(file,'r') as in_f:
                for line in in_f.readlines():
                    info = line.split(' ',1)
                    if info[0] == '__label__NULL':
                        null_f.write(line)
                    else:
                        out_f.write(line)

#if __name__ == "__main__":
#    file = "./procressed_data_new/procressed_daodao_zh_shuffle.txt"
#    file = "daodao_zh_shuffle.txt"
#    out_file = "./procressed_data_new/procressed_daodao_zh_shuffle.txt"
#    repls = punctuation_dict()
#    procress_zh_data(file,out_file)
#    comment_sents_zh_id(file,out_file,flag=False)
#    comment_sents_zh_id(file,out_file)

#    infile = "/data/caozhaojun/true_procress_data/procressed_data_new/test_result/add_zh_test_data.txt"
#    outfile = "/data/caozhaojun/true_procress_data/procressed_data_new/test_result/add_zh_test_clean.txt"
#    make_test_data_for_evaluate(infile,outfile,lang="zh")

#    file = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_zh_test.txt"
#    out_file = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_zh_test_null.txt"
#    create_null_test_data(file,out_file)

#    file = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en.txt"
#    out_file = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_not_null.txt"
#    out_null = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_null.txt"
#    remove_null(file,out_file,out_null)
