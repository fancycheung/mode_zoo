import argparse
import os
import sys
from  procress_data import comment_sents_en_line,comment_sents_zh_line
import re 
from collections import defaultdict,OrderedDict,Counter
from random import shuffle
import numpy as np

def process_each_comment(comment,save_dir,lang="en"):
    '''处理输入评论，一个评论处理、分句，从而按句输入模型'''
    if lang == "en":
        line_list = comment_sents_en_line(comment)
    elif lang == "zh":
        line_list = comment_sents_zh_line(comment)
    with open(save_dir,'w') as f:
        for line in line_list:
            f.write(line)


def sample_weighted(infile,outfile):
    '''对输入评论按标签数量采样，少的标签多采样'''
    pattern = re.compile(r"(\b__label__(\S+))")
    count = Counter()
    with open(infile,'r') as in_f:
        lines = in_f.readlines()
        for line in lines:
            find = re.findall(pattern,line)
            label = [item[1] for item in find]
            count.update(label)
        most = count.most_common(2)[1][1]
        print(count)
        return
        with open(outfile,"w") as out_f:
            cache_list = []
            for line in lines:
                find = re.findall(pattern,line)
                label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                if len(comment) > 2:
                    values = [5] # 设定最多采样5次
                    for item in label:
                        value = count.get(item)
                        if value is not None:
                            values.append(int(np.floor(most / value)))
                    if min(values) >= 1:
                        for i in range(min(values)):
                            cache_list.append(line)
                    elif min(values) < 1:
                        cache_list.append(line)
            shuffle(cache_list)
            for line in cache_list:
                out_f.write(line)

def file_for_view(infile,outfile):
    '''格式化输出结果，便于查看'''
    pattern = re.compile(r"(\b__label__(\S+))")
    with open(infile,'r') as in_f:
        with open(outfile,"w") as out_f:
            lines = in_f.readlines()
            for line in lines:
                find = re.findall(pattern,line)
                label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                label = line[:len(label_len)].strip()
                out_f.write(comment+"\n"+"\t"+label+"\n\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inf",default="")
    parser.add_argument("--out",default="./data/")
    parser.add_argument("--lang",default="en")
    parser.add_argument("--mode")
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if args.mode == 1:
        process_each_comment(args.inf,args.out+"split_comment.txt",args.lang)
    elif args.mode == 2:
        sample_weighted(args.inf,args.out+"ft_%s_weighted.train" % args.lang)
    elif args.mode == 3:
        file_for_view(args.inf,args.inf+"_view")



    



