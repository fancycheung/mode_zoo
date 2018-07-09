#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:39:04 2018

@author: miaoji
"""

import json
from collections import defaultdict
import os
import json
import re 
from tqdm import tqdm

class Label():
    def __init__(self,id):
        self.id = id
        self.labelName = 'NULL'
        self.domain = 'NULL'
        self.domainName = 'NULL'
        self.emotion = 0
        self.display = 1


def readLabel(label_file = 'label.mode.txt'):

    labels = {}
    for line in open(label_file,'r'):
        if line.strip() == '' or line[0] == '#':
            continue
        infos = line.strip().split('\t')
        id,labelName,emotion,domainName,domain,display = infos[0],infos[1],int(infos[2]),infos[5],infos[6],int(infos[7])
        label = Label(id)
        label.labelName = labelName
        label.domain = domain
        label.domainName = domainName
        label.emotion = emotion
        label.display = display
    
        labels[label.id] = label

    return labels

LabelD = readLabel()
punct = ['！','？','：','。','，','?','!',':','.',',','#']

         
def pack_cpp(input):
    '''打包为c++格式'''
    input = input.replace('\n','.')

    for symbol in punct:
        input = input.replace(symbol,'_' + symbol)

    input = input.replace(' ','_') + '_'
    input = input.replace('__','_')

    return input


def strQ2B(ustring):
    '''全角转半角'''
    rstring = ''
    try:
        ustring_utf8 = ustring
    except:
        return ustring

    for uchar in ustring_utf8:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code > 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)

    return rstring


def strB2Q(ustring):
    '''半角转全角'''
    rstring = ""
    try:
        ustring_utf8 = ustring.decode('utf-8')
    except:
        return ustring

    for uchar in ustring_utf8:
        inside_code = ord(uchar)
        if inside_code == 32:
            inside_code == 12288
        elif inside_code >= 32 and inside_code <= 126:
            inside_code += 65248

        rstring += chr(inside_code)

    return rstring.encode('utf-8')

def pack_infile_line(line):
    comment = line.strip().replace(':',' ').replace('\r',' ')
    comment_packed = pack_cpp(comment)
    return comment_packed

def pack_infile(infile,outfile):
    '''对文件打包'''
    with open(outfile,'a+') as w:
        with open(infile,'r') as f:
            for line in tqdm(f.readlines()):
                if line.strip() == "": continue
                try:
                    #comment = line.strip().replace(':',' ').replace('\r',' ')
                    #comment_packed = pack_cpp(comment)
                    comment_packed = pack_infile_line(line)
                    w.write(comment_packed+'\n')
                except Exception as e:
                    print(str(e))
                    continue


def unpack_line(line):
    '''按行解包'''
    return strQ2B(line).replace('_',' ').strip()


def handle_json(json_f):
    '''解析json'''
    dict_f = json.loads(json_f)
    if dict_f['tags'] in ['null',None,[None]]:
        result_label = 'NULL'
    else:
        tags = []
        for inner_dict in dict_f['tags']:
            for hit_tag in inner_dict.values():
                tag = hit_tag['tag'].split('|')
                for hit in hit_tag['hit']:
                    for t in tag:
                        tags.append((unpack_line(hit),t))
        if len(tags) == 0:
            result_label = 'NULL'
        else:
            result_label = dict()
            for item in tags:
                tmp_dict = {}
                tmp_dict['hit'] = item[0]
                tmp_dict['label'] = LabelD[item[1]].labelName
                result_label[item[1]] = tmp_dict
                            
    return result_label
    

def unpack_outfile(in_file,out_file):
    '''对文档解包'''
    n = 0
    with open(out_file,'a+') as w:
        with open(in_file,'r') as f:
            for line in tqdm(f.readlines()):
                n += 1
                comment_tags = line.strip().split('\t')
                if len(comment_tags) == 2: 
                    comment,json_tag = comment_tags
                    result_label = handle_json(json_tag)
                    str_label = str(result_label)
                    comment_unpack = comment.replace('_',' ').strip()
                    try:
                       # comment_remove_id = comment_unpack.split(' ',1)[1]
                        comment_remove_id = comment_unpack
                        result_line = '\t'.join([str_label,comment_remove_id]) 
                        w.write(result_line+'\n')
                    except:
                        print(str_label,comment_unpack)
       

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

         
def create_ft_small_test_data(file,out_file):
    with open(out_file,'w') as out_f:
        with open(file,'r') as in_f:
            info_list = []
            for line in in_f.readlines():
                infos = line.split(' ',1)
                info_list.append(infos[1])
#	    info_set = set(info_list)
            info_set = set(info_list)
            for item in info_set:
                out_f.write(item)

def create_null_test_data(file,out_file):
    with open(out_file,'w') as out_f:
        with open(file,'r') as f:
            pattern = re.compile(r"(\b__label__(\S+))")
            for line in f.readlines():
                find = re.findall(pattern,line)
                label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                comment = comment +'\n'
                out_f.write(comment)

if __name__ == "__main__":
#    pack_infile('/data/caozhaojun/true_procress_data/procressed_data_new/procressed_daodao_zh_shuffle.txt','./procressed_data_new/procressed_daodao_zh_shuffle_packed.txt')
#    pack_infile('/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_zh_test_null.txt','./procressed_data_new/multi_ft_format_zh_test_null_packed.txt')
#    in_file = '/data/caozhaojun/zh_result.txt'
#    out_file = './procressed_data_new/test_zh_result_unpack.txt'
#    unpack_outfile(in_file,out_file)

#    in_file = './procressed_data_new/procressed_daodao_zh_shuffle_packed_result.txt'
#    out_file = './procressed_data_new/procressed_daodao_zh_shuffle_packed_result_unpack.txt'
#    unpack_outfile(in_file,out_file)

#    file = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_zh.txt"
#    out_file = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_zh_not_null.txt"
#    out_null = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_zh_null.txt"
#    remove_null(file,out_file,out_null)

