#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:50:34 2018

@author: miaoji
"""

import langid
import os
from tqdm import tqdm
import threading

def list_in_dir(indir):
    all_in_dir = [indir+i for i in os.listdir(indir)]
    return all_in_dir

def transform(in_dir,output_f_en,output_f_zh):
    '''
        判断语言，归类相同语言
    '''
    all_in_dir = list_in_dir(in_dir)
    print(len(all_in_dir))

    tmp_id = 0
    tmp_id_l = 0
    en_id_l = 0
    zh_id_l = 0

    with open(output_f_en,'a+') as out_f_en:
        with open(output_f_zh,'a+') as out_f_zh:

            for tem_in in all_in_dir:
                tmp_id += 1

                if tmp_id % 100 == 0:
                    print(tem_in)
                    print(tmp_id/len(all_in_dir))

                with open(tem_in,'r',errors='ignore') as in_f:

                    for each_line in in_f.readlines():
                        line =each_line
                        tmp_id_l += 1
                        line_tuple = langid.classify(line)
                        if line_tuple[0] == 'en':
                            out_f_en.write(line)
                            en_id_l += 1
                        if line_tuple[0] == 'zh':
                            out_f_zh.write(line)
                            #out_f_en.write(str(tem_in)+' ZH '+line+'\n')
                            zh_id_l += 1
    return tmp_id_l,en_id_l,zh_id_l


def add_langid_old(in_dir,out_file):
    
    all_in_dir = list_in_dir(in_dir)
    print(len(all_in_dir))
    with open(out_file,'a+') as out_f:
        for tem_in in tqdm(all_in_dir):
            with open(tem_in,'r',errors='ignore') as in_f:
                for each_line in in_f.readlines():
                    line_tuple = langid.classify(each_line)
                    out_line = line_tuple[0]+' '+each_line
                    out_f.write(out_line)

def add_langid_one_file(in_dir,out_file):
    with open(out_file,'a+') as out_f:
        with open(in_dir,'r',errors='ignore') as in_f:
            for each_line in tqdm(in_f.readlines()):
                line_tuple = langid.classify(each_line)
                out_line = line_tuple[0]+' '+each_line
                out_f.write(out_line)
            
    
if __name__ == '__main__':
    os.chdir('/data/caozhaojun/true_procress_data/')
    in_dir = './merge_source/eu_booking'
    out_file = './merge_file/eu_booking_another'
    add_langid_one_file(in_dir,out_file)
    '''
#    file_en= 'daodao_en.txt'
#    file_zh = 'daodao_zh.txt'
#    id_l,en_l,zh_l = transform(in_dir,file_en,file_zh)
#    print(id_l,en_l,zh_l)
    in_base_dir = './hotel_comment_by_source/'
    out_base_dir = './merge_source/'
#    add_langid(in_dir,out_file)
    file_list = ['eu_booking','hotelclub','hotels','na_booking']
    threads = []
    in_dire = []
    out_file = []
    for i in range(len(file_list)):
        in_dire.append(in_base_dir + file_list[i] + '/')
        out_file.append(out_base_dir + file_list[i])
    print(in_dire)
    print(out_file)
    for i in range(len(file_list)):
        print(i,file_list[i]) 
        merge_same_dir_hotel(in_dire[i],out_file[i])
    
#    add_langid(in_dir[0],out_file[0])
    
    
    thing0 = threading.Thread(target=add_langid,args=(in_dir[0],out_file[0]))
    thing1 = threading.Thread(target=add_langid,args=(in_dir[1],out_file[1]))
    thing2 = threading.Thread(target=add_langid,args=(in_dir[2],out_file[2]))
    thing3 = threading.Thread(target=add_langid,args=(in_dir[3],out_file[3]))
    threads.append(thing0)
    threads.append(thing1)
    threads.append(thing2)
    threads.append(thing3)

    print("start")    
    for thing in threads:
        thing.setDaemon(True)
        thing.start()
    
    thing0.join()
    thing1.join()
    thing2.join()
    thing3.join()
    '''
    print("end")
    
