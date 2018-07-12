#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 09:29:49 2018

@author: miaoji
"""

import argparse
import os
import sys

import  pattern_module_utils as pm
import procress_data as pre

def clear_text_func(file,lang,outfile,write=True):
    clear_texts = []
    with open(file,"r") as f:
        line_id = 0
        all_line = f.readlines()
        for line in all_line:
            line_id += 1
            if lang == "zh":
                line_list = pre.comment_sents_zh_id_line(line,line_id)

                clear_texts.extend(line_list)
            elif lang == "en":
                line_list = pre.comment_sents_en_id_line(line,line_id)
                clear_texts.extend(line_list)
    if write == True:
        with open(outfile,"w") as out_f:
            for line in clear_texts:
                out_f.write(line+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text",default="")
    parser.add_argument("--save_dir",default="./data")
    parser.add_argument("--lang",default="en")
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #clear_text_func(file,lang,outfile,write=True)   




