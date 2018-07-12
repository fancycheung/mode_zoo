#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:10:41 2018

@author: miaoji
"""

import re
import argparse

def evaluate(predict_label,true_label):
    y_ = set(predict_label)
    y = set(true_label)
    accuracy = len(y & y_) / (len(y|y_))
    precision = len(y & y_) / (len(y_))
    recall = len(y & y_) / (len(y))
    f1_score = 2.0 * precision * recall / (precision + recall + 0.000001)
#    if counter % 3 == 0:
#        print(counter,":",y,y_)
    return accuracy, precision, recall, f1_score

def load_data(file1,file2,gate):
    with open(file1,'r') as f1:
        with open(file2,'r') as f2:
            pattern = re.compile(r"(\b__label__(\S+))")
            eval_counter,eval_acc,eval_p,eval_r,eval_f1_score = 0,0.0,0.0,0.0,0.0
            f1_lines = f1.readlines()
            f2_lines = f2.readlines()
            
            assert len(f1_lines) == len(f2_lines)
            for i in range(len(f1_lines)):
                line_1 = f1_lines[i]
                line_2 = f2_lines[i]
                
                find_1 = re.findall(pattern,line_1)
                label_1 = [item[1] for item in find_1]
                
                find_2 = re.findall(pattern,line_2)
                label_2 = [item[1] for item in find_2]
                
                acc,precision,recall,f1_score = evaluate(label_1,label_2)
                eval_counter,eval_acc,eval_p,eval_r,eval_f1_score = eval_counter+1,eval_acc+acc,eval_p+precision,eval_r+recall,eval_f1_score+f1_score

            print("gate:%.2f eval_acc:%f eval_p:%f eval_r:%f eval_f1:%f" % (gate,eval_acc/float(eval_counter),eval_p/float(eval_counter),eval_r/float(eval_counter),eval_f1_score/float(eval_counter)))

if __name__ == "__main__":
   # file1 = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_test_predict.txt"
   # file2 = "/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_test.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate",type=float)
    parser.add_argument("--lang")
    args = parser.parse_args()
    lang = args.lang
    gate = args.gate
    print("add input file yourself!")
    #load_data(file2,file1,gate=gate)
    '''
    print("fasttext")
    print(lang,gate)
    langs = ["en","zh"]
    gates = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
    for lang in langs:
        print("+"*30+lang+"+"*30)
        for gate in gates:
            #print("--"*30)
            file1 = "./data/ft_%s.eval" % lang
            file2_1 = "./data/predict_ft_%s_%.2f" % (lang,gate)
            file2_2 = "./data/predict_ft_%s_weighted_%.2f" % (lang,gate)
            load_data(file2_1,file1,gate=gate)
    '''
   #         load_data(file2_2,file1,gate=gate)
#    print("pattern")
#    file2_2 = "/data/caoizhaojun/true_procress_data/procressed_data_new/test_result/multi_ft_format_%s_test_pattern_predict.txt" % lang
#    load_data(file2_2,file1)
#    print("fasttext-pattern")
#    load_data(file2_1,file2_2)

