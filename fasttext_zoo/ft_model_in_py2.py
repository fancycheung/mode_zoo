# -*- coding:utf-8 -*-

import fasttext
import uniout
import codecs
import sys
from tqdm import tqdm
import re
import argparse

reload(sys)
sys.setdefaultencoding('utf8')


def predict_format_result_multi(file,out_file,gate=0.1):
    with codecs.open(out_file,'w') as out_f:
        with codecs.open(file,'r','utf-8') as f:
            pattern = re.compile(r"(\b__label__(\S+))")
            for line in f.readlines():
                find = re.findall(pattern,line)
                #label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                comment = comment + '\n'
                predict_label = classifier.predict_proba([comment],5)
                new_label = []
                for label,prob in predict_label[0]:
                    if prob >= gate:
                        new_label.append(label.encode())
                if len(new_label) < 1:
#                    new_label.append(predict_label[0][0][0])
                    new_label.append("NULL")
                if len(new_label) > 1:
                    if 'NULL' in new_label:
                        new_label.remove('NULL')

                predict_label = ' '.join(["__label__"+x for x in new_label])
                out_f.write(predict_label+' '+comment)

def predict_format_result_multi_for_view(file,out_file,gate=0.05):
    with codecs.open(out_file,'w') as out_f:
        with codecs.open(file,'r','utf-8') as f:
            pattern = re.compile(r"(\b__label__(\S+))")
            for line in tqdm(f.readlines()):
                find = re.findall(pattern,line)
                label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                comment = comment + '\n'
                predict_label = classifier.predict_proba([comment],5)
                new_label = []
                for label,prob in predict_label[0]:
                    if prob >= gate:
                        new_label.append(label.encode())
                if len(new_label) < 1:
                    new_label.append("NULL")
                if len(new_label) > 1:
                    if 'NULL' in new_label:
                        new_label.remove('NULL')

                predict_label = ' '.join(["__label__"+x for x in new_label])
                out_f.write(comment+"\t"+predict_label+"\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate',default=None,type=float)
    parser.add_argument('--lang', default="zh")
    parser.add_argument('--train',default=0,type=int)
    parser.add_argument('--view',default=0,type=int)
    parser.add_argument('--lr',default=0.2,type=float)
    parser.add_argument('--lr_update',default=50,type=int)
    args = parser.parse_args()
    lang = args.lang
    gate = args.gate
    weight = args.weight
    view = args.view
    if args.train == 1:
        print(args)
        file = "./data/ft_%s.train" % lang
        if lang == "zh":
            if weight:
                model_path = "./model/ft_format_zh_200d_weight"
            else:
                model_path = "./model/ft_format_zh_200d"
            vec_path = './data/daodao_zh_word2vec.txt'
            classifier = fasttext.supervised(file,model_path,label_prefix='__label__',epoch=50,dim=200,min_count=5,pretrained_vectors=vec_path,lr=args.lr,lr_update_rate=args.lr_update)
        elif lang == "en":
            model_path = "./model/ft_format_en_300d"
            vec_path = "./data/multi_ft_en_train_cbow_300d.vec"
            classifier = fasttext.supervised(file,model_path,label_prefix='__label__',epoch=50,dim=300,min_count=5,pretrained_vectors=vec_path,lr=args.lr,lr_update_rate=args.lr_update)
        print("train end!")

    if args.train == 0:
        if lang == 'zh':
            model_path = "./model/ft_format_zh_200d.bin"
        elif lang == "en":
            model_path = "./model/ft_format_en_300d.bin"
        classifier = fasttext.load_model(model_path,label_prefix='__label__',encoding='utf-8')
        test_data = "./data/ft_%s.eval" % lang
        out_file = "./data/predict_ft_%s_%.2f" % (lang,gate)
        if not view:
            predict_format_result_multi(test_data,out_file,gate=gate)
        if view:
            out_file = "./data/predict_ft_%s_%.2f_view" % (lang,gate)
            predict_format_result_multi_for_view(test_data,out_file,gate)






