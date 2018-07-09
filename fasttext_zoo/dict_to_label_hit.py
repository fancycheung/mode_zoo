#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:52:36 2018

@author: miaoji
"""
from tqdm import tqdm
from procress_data import clean_str
import jieba
import re
import langid

def extract_dict(file,out_file):
    out_list = []
    n = 0
    with open(out_file,'w') as w:
        with open(file,'r') as f:
            for line in tqdm(f.readlines()):
                n += 1
                dict_comment = line.split('\t')
                label_str,_ = dict_comment
                if label_str == 'NULL':
                    continue
                else:
                    label_dict = eval(label_str)
                    for k,v in label_dict.items():
                        tmp_dict = {}
                        tmp_dict[k] = v['hit']
                        out_list.append(tmp_dict)
        out_list_str = str(out_list)   
        w.write(out_list_str)

def clean_seg_coment(line):
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

def extract_dict_ft_format_zh(file,out_file):
    '''multi-label zh'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            for line in tqdm(in_f.readlines()):
                sent = line.split('\t',1)
                label_str,sent_comment = sent
                sent_comment = sent_comment.replace("\n",'')
                #sent_comment = clean_str(sent_comment)   # use for en
                sent_comment = sent_comment.replace(" ","") #zh
                seg_line = jieba.cut(sent_comment)#zh
                sent_comment = clean_seg_coment(seg_line)#zh
                label = []
                if label_str == 'NULL':
                    final_line = '__label__' + 'NULL' + ' ' + sent_comment + '\n'
                    out_f.write(final_line)
                else:
                    label_dict = eval(label_str)
                    flag = 0 #zh
                    for k,v in label_dict.items():
                        hit_lang = langid.classify(v["hit"])[0] # zh
                        if hit_lang != "zh": #zh
                            flag = 1  # zh
                        label.append('__label__'+v['label'])
                    tmp_label = ' '.join([x for x in label])
                    final_line = tmp_label  + ' ' + sent_comment + '\n'
                    if flag == 0: #zh
                        out_f.write(final_line)

def extract_dict_ft_format_en(file,out_file):
    '''multi-label en'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            for line in tqdm(in_f.readlines()):
                sent = line.split('\t',1)
                label_str,sent_comment = sent
                sent_comment = sent_comment.replace("\n",'')
                sent_comment = clean_str(sent_comment)   # use for en
                label = []
                if label_str == 'NULL':
                    final_line = '__label__' + 'NULL' + ' ' + sent_comment + '\n'
                    out_f.write(final_line)
                else:
                    label_dict = eval(label_str)
                    for k,v in label_dict.items():
                        label.append('__label__'+v['label'])
                    tmp_label = ' '.join([x for x in label])
                    final_line = tmp_label  + ' ' + sent_comment + '\n'
                    out_f.write(final_line)

def extract_dict_ft_format_tmp(file,out_file):
    '''multi-label'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            for line in tqdm(in_f.readlines()):
                sent = line.split('\t',1)
                label_str,sent_comment = sent
                sent_comment = sent_comment.replace("\n",'')
                sent_comment = clean_str(sent_comment)
                label = []
                if label_str == 'NULL':
                    final_line = '__label__' + 'NULL'+'\n'
                    out_f.write(final_line)
                else:
                    label_dict = eval(label_str)
                    for k,v in label_dict.items():
                        label.append('__label__'+v['label'])
                    tmp_label = ' '.join([x for x in label])
                    final_line = tmp_label  + '\n'
                    out_f.write(final_line)

def extract_dict_ft_format_nolabel(file,out_file):
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            for line in tqdm(in_f.readlines()):
                sent = line.split('\t',1)
                label_str,sent_comment = sent
                sent_comment = sent_comment.replace("\n",'')
                sent_comment = clean_str(sent_comment)
                out_f.write(sent_comment+'\n')

def classify_none_line_and_other(file,out_file1,out_file2):
    with open(out_file1,'w') as out_f1:
        with open(out_file2,'w') as out_f2:
            with open(file,'r') as in_f:
                for line in tqdm(in_f.readlines()):
                    label1,false_comment = line.split(' ',1)
                    if label1 == '__label__NULL':
                        out_f1.write(line)
                    else:
                        out_f2.write(line)

def extract_dict_textcnn_format(file,out_file):
    '''textcnn format'''
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            for line in tqdm(in_f.readlines()):
                sent = line.split('\t',1)
                label_str,sent_comment = sent
                sent_comment = sent_comment.replace("\n",'')
                sent_comment = clean_str(sent_comment)
                label = []
                if label_str == 'NULL':
                    final_line = 'NULL' + ' ' + sent_comment + '\n'
                else:
                    label_dict = eval(label_str)
                    for k,v in label_dict.items():
                        label.append(v['label'])
                    final_line = '_'.join(label) + ' ' + sent_comment + '\n'
                out_f.write(final_line)


if __name__ == '__main__':
    
#    file = './procressed_data_new/test_zh_result_unpack.txt'
#    out_file = './procressed_data_new/test_zh_result_unpack_ft.txt'    
#    extract_dict_ft_format_tmp(file,out_file)

    file = "./procressed_data_new/procressed_daodao_zh_shuffle_packed_result_unpack.txt"
    out_file = "./procressed_data_new/multi_ft_format_zh.txt"
#    extract_dict_textcnn_format(file,out_file)
#    out_file = "./procressed_data_new/ft_format_en.txt"
    extract_dict_ft_format_zh(file,out_file)

#    out_file = "./procressed_data_new/ft_format_en_nolabel.txt"
#    extract_dict_ft_format_nolabel(file,out_file)
#    file = '/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en.txt'
#    out_file1 = '/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_not_none.txt'
#    out_file2 = '/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_none.txt'    
#    classify_none_line_and_other(file,out_file1,out_file2)




