import argparse
import os
import sys
import re
import numpy as np

def file_for_view(file1,file2,outfile):
    pattern = re.compile(r"(\b__label__(\S+))")
    with open(file1,'r') as f1:
        texts1 = f1.readlines()
    with open(file2,'r') as f2:
        texts2 = f2.readlines()
    with open(outfile,'w') as out:
        assert len(texts1) == len(texts2)
        for i in range(len(texts1)):
            find = re.findall(pattern,texts1[i])
            label_len = ' '.join([item[0] for item in find])
            comment = texts1[i][len(label_len):].strip()
            label = texts2[i]
            out.write(comment+"\n\t"+label+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test")
    parser.add_argument("--label")
    parser.add_argument("--out")
    args = parser.parse_args()

    file_for_view(args.test,args.label,args.out)





