#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:25:19 2018

@author: miaoji
"""
import nltk.tokenize as nt
from textblob import TextBlob
import time

start_time = time.time()

in_file = open("/data/zhangbin/caozhaojun/true_procress_data/daodao_en.txt",'r')
out_file = open("handle_daodao_en.txt",'a+')
tokenizer = nt.TweetTokenizer()

line_id = 0
for line in in_file.readlines():
    line_id += 1
    if line_id % 1000 == 0:
        print(line_id)
    correct_line = TextBlob(line.lower().replace('...',' ').strip())#.correct()
    token_line = correct_line.tokenize(tokenizer)
    final_line = ' '.join([word for word in token_line])
    out_file.write(final_line+'\n')
in_file.close()
out_file.close()    

end_time = time.time()
print(float(end_time - start_time))






