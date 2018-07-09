#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:15:53 2018

@author: miaoji
"""
from random import shuffle
from tqdm import tqdm
from collections import defaultdict,OrderedDict,Counter
import os
import pickle
import re
import numpy as np

def merge_result_for_view(file1,file2,out_file):
    with open(out_file,'w') as out_f:
        with open(file1,'r') as f1:
           with open(file2,'r') as f2:
               all_f1 = f1.readlines()
               all_f2 = f2.readlines()
               assert len(all_f1) == len(all_f1)
               for i in range(len(all_f1)):
                  # final_line = all_f1[i] +"\t"+ all_f2[i] + "\n"
                   final_line = all_f2[i].replace("\n","")+" "+all_f1[i]
                   out_f.write(final_line)

def merge_shuffle_train_data(file1,file2,out_file,split_n=100000):
    with open(out_file,'w') as out_f:
        with open(file1,'r') as f1:
           with open(file2,'r') as f2:
               all_f1 = f1.readlines()
               all_f2 = f2.readlines()
               all_f = all_f1 + all_f2
               shuffle(all_f)
               for item in all_f:
                   out_f.write(item)

def shuffle_train_data(infile,outfile):
    with open(infile,'r') as in_f:
        all_in = in_f.readlines() 
        shuffle(all_in)
        with open(outfile,'w') as out_f:
            for line in all_in:
                out_f.write(line)
        

def merge_same_source_ignore_null_line(outfile,file1,file2=None):
    with open(outfile,'w') as out_f:
        with open(file1,'r') as in_f1:
            lines_1 = in_f1.readlines()

            if file2 != None:
                with open(file2,'r') as in_f2:
                    lines_2 = in_f2.readlines()
                    lines = lines_1 + lines_2
                    for line in tqdm(lines):
                        split_line = line.split(' ',1)
                        if len(split_line[1]) < 2:
                            continue
                        else:
                            out_f.write(line)
            else:
                for line in tqdm(lines_1):
                    split_line = line.split(' ',1)
                    if len(split_line[1]) < 2:
                        continue
                    else:
                        out_f.write(line)
 
def save_split_lang(infile,outdir,pkl_dir):
    with open(infile,'r') as in_f:
        tmp_list = []
        for line in tqdm(in_f.readlines()):
            key,comment = line.split(' ',1)
            tmp_list.append(key)
            file_dir = outdir + key + '.txt'
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            with open(file_dir,'a') as out_f:
                out_f.write(comment)
        lang_counter = Counter(tmp_list)
        
        print('start pickle')
        with open(pkl_dir,'wb') as pkl:
            pickle.dump(lang_counter,pkl)
        
        print('pickle end')


def procress_zh(ss):  
    s2=re.sub(r'[a-zA-Z0-9\.]+','',ss)    
    s3=re.sub('\s+','',s2)
    return s3.strip()

def stat_word_num(infile,pkl_dir,lang):
    with open(infile,'r') as in_f:
        comment_len = []
        if lang == 'en':
            for line in tqdm(in_f.readlines()):
                line_split = line.strip().split(' ')
                comment_len.append(len(line_split))
        if lang == 'zh':
            for line in tqdm(in_f.readlines()):
                line_clear = procress_zh(line)
                comment_len.append(len(line_clear))
        comment_counter = Counter(comment_len)
        print('start pickle')
        with open(pkl_dir,'wb') as pkl:
            pickle.dump(comment_counter,pkl)
        print('pickle end')

def sample_weighted(infile,outfile):
    pattern = re.compile(r"(\b__label__(\S+))")
    count = Counter()
    with open(infile,'r') as in_f:
        lines = in_f.readlines()
        for line in lines:
            find = re.findall(pattern,line)
            label = [item[1] for item in find]
            count.update(label)
        most = count.most_common(1)[0][1]
        print(count)
        with open(outfile,"w") as out_f:
            for line in lines:
                find = re.findall(pattern,line)
                label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                if len(comment) > 2:
                    values = [10000]
                    for item in label:
                        value = count.get(item)
                        if value is not None:
                            values.append(int(np.floor(most / value)))
                    for i in range(min(values)):
                        out_f.write(line)
        

if __name__ == '__main__':
#    name_list = ['hotelclub','daodao','hotels','booking']
#    langs = ['zh','en']
#    for name in name_list:
#        for lang in langs:
#            infile = './merge_source_id/split_lang_%s/%s.txt' % (name,lang)
#            pkl_dir = './merge_source_id/comment_len_pickle/%s_%s.pickle' % (name,lang)
#            stat_word_num(infile,pkl_dir,lang)

#    infile = './merge_source_id/merge_%s' % (name)
#    outdir = './merge_source_id/split_lang_%s/' % (name)
#    pkl_dir = './merge_source_id/cache_pickle/%s.pickle' % (name)
#    save_split_lang(infile,outdir,pkl_dir)

#    file1 = './procressed_data/ft_format_data_tmp_train_nonull.txt'
#    file2 = './procressed_data/train_null_with_hard_null_high_label_99_20_train_nonull.txt'
#    file3 = './procressed_data/train_null_with_hard_null_high_label_99_20_train_null.txt'
#    out_file = './procressed_data/ft_format_data_tmp_merge_train_1.txt'
#    merge_shuffle_train_data(file1,file2,file3,out_file)

#    file1 = './merge_source_id/hotelclub'
#    file2 = './merge_source_id/na_booking'
#    outfile = './merge_source_id/merge_hotelclub'
#    merge_same_source_ignore_null_line(outfile,file1,file2=None)
#    file1 = '/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_test_null.txt'
#    file2 = '/data/caozhaojun/true_procress_data/procressed_data_new/test_en_result_unpack_label.txt'
#    file2 = '/data/caozhaojun/textcnn_model/result.txt'
#    out_file = '/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_test_pattern_predict.txt'
#    merge_shuffle_train_data(file1,file2,out_file)
#    merge_result_for_view(file1,file2,out_file)

#    infile = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_not_null.txt"
#    outfile = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_sample.txt"
#    sample_weighted(infile,outfile)


    infile = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_sample.txt"
    outfile = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_train.txt"
    shuffle_train_data(infile,outfile)


