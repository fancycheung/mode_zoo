# -*- coding:utf-8 -*-

import fasttext
import uniout
import codecs
import sys  
from tqdm import tqdm
import re
 
reload(sys)  
sys.setdefaultencoding('utf8')  

def save_result():
    '''以固定格式保存测试结果'''
    with codecs.open(out_file,'w') as out_f:
        with codecs.open(test_data,'r') as f:
            for line in f.readlines():
                line = line.replace('\n','').strip()
		tmp_line = []
		tmp_line.append(str(line))
		line = tmp_line
                labels = classifier.predict_proba(line,3)
		label_probs = []
		for item in labels[0]:
		    label,prob = item
		    label_prob = label+':'+str(prob)
		    label_probs.append(label_prob)
		label_probs = '__'.join(label_probs)
                final_line = label_probs + ' ' + line[0] + '\n'
                out_f.write(final_line)
        
def print_for_view(file):
    with open(file,'r') as f:
        for i,line in enumerate(f.readlines()):
            if i < 10:
                print(line)
            else:
                break


def predict_format_result(file,out_file):
    with codecs.open(out_file,'w') as out_f:
        with codecs.open(file,'r','utf-8') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                old_label = split_line[0]
                x = split_line[1]
                predict_label = classifier.predict_proba([x],3)
                new_label = []
                for label,prob in predict_label[0]:
                    if prob >= 0.1:
                        new_label.append(label+':'+str(prob))
                predict_label = '_'.join(new_label)
                out_f.write(x+'\t'+old_label+'\n'+'\t'+'new_label:'+predict_label+'\n\n')

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
#                if len(new_label) > 1:
#                    if 'NULL' in new_label:
#                        new_label.remove('NULL')

                predict_label = ' '.join(["__label__"+x for x in new_label])
                out_f.write(predict_label+' '+comment)

def create_null_test_data(file,out_file):
    with open(out_file,'w') as out_f:
        with open(file,'r') as f:
            pattern = re.compile(r"(\b__label__(\S+))")
            for line in f.readlines():
                find = re.findall(pattern,line)
                #label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()
                comment = comment +'\n'
                out_f.write(comment)

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
                out_f.write(comment+"\n"+"\t"+predict_label+"\n\n")

def create_hard_null_and_high_label(file,out_file,for_view=False):
    with open(out_file,'w') as out_f:
        with open(file,'r') as f:
            pattern = re.compile(r"(\b__label__(\S+))")
            for line in tqdm(f.readlines()):
                #split_line = line.split(' ',1)
                #label = split_line[0].split('__label__')[1]
                #comment = split_line[1]

                find = re.findall(pattern,line)
                label = [item[1] for item in find]
                label_len = ' '.join([item[0] for item in find])
                comment = line[len(label_len):].strip()                
                comment = comment +'\n'
#                print("comment",comment)
#                try:
                predict_label = classifier.predict_proba([comment],8)
#                except:
#                    print("comment: ",comment)
#                    break

                pre_label = []
                pre_prob = []
                for label,prob in predict_label[0]:
                    if prob >= 0.25:
                        pre_label.append(label)
                        pre_prob.append(prob)
                
                if for_view == True:
                    if 'NULL' in pre_label:
                        tmp_index = pre_label.index('NULL')
                        if pre_prob[tmp_index] > 0.99:
                            final_label = 'NULL'
                            out_f.write(final_label+' '+comment)
                        elif len(pre_label) >= 2:
                            pre_label.remove('NULL')
                            pre_prob.pop(tmp_index)
                            pre_label = list(map(lambda x,y:x+':'+str(y),pre_label,pre_prob))
                            final_label = '__'.join(pre_label)
                            out_f.write(final_label+' '+comment)
                    
                    elif len(pre_label) >= 1:
                        pre_label = list(map(lambda x,y:x+':'+str(y),pre_label,pre_prob))
                        final_label = '__'.join(pre_label)
                        out_f.write(final_label+' '+comment)
                
                else:
                    if 'NULL' in pre_label:
                        tmp_index = pre_label.index('NULL')
                        if pre_prob[tmp_index] > 0.99:
                            final_label = '__label__NULL'
                            out_f.write(final_label+' '+comment)
                        elif len(pre_label) >= 2:
                            pre_label.remove('NULL')
                            pre_prob.pop(tmp_index)
                            label_list = []
                            for i in xrange(len(pre_label)):
                                label_list.append('__label__'+pre_label[i])
                            final_label = ' '.join(label_list)
                            out_f.write(final_label+' '+comment)
                    
                    elif len(pre_label) >= 1:
                        label_list = []
                        for i in xrange(len(pre_label)):
                            label_list.append('__label__'+pre_label[i])
                        final_label = ' '.join(label_list)
                        out_f.write(final_label+' '+comment)



def do_eval(file,out_file):
    with codecs.open(file,'r','utf-8') as data_f:
        lines = data_f.readlines()
    labels_right = []
    texts = []
    for line in lines:
        labels_right.append(line.split(" ",1)[0].strip().replace("__label__",""))
        texts.append(line.split(" ",1)[1])
    
    labels_predict = [e[0] for e in classifier.predict(texts)]
    
    text_labels = list(set(labels_right))  
    text_predict_labels = list(set(labels_predict))  
    
    A = dict.fromkeys(text_labels,0)  
    B = dict.fromkeys(text_labels,0)   
    C = dict.fromkeys(text_predict_labels,0) 

    for i in range(0,len(labels_right)):  
        B[labels_right[i]] += 1  
        C[labels_predict[i]] += 1  
        if labels_right[i] == labels_predict[i]:  
            A[labels_right[i]] += 1  
            
    with open(out_file,'w') as out_f:
        for key in B:  
            try:
                p = float(A[key]) / float(B[key])  
                r = float(A[key]) / float(C[key])  
                f = p * r * 2 / (p + r)  
                out_str = "%s:\t precision:%f\t recall:%f\t f1:%f" % (key,p,r,f)
            except:
                print("zero!")
                continue
            out_f.write(out_str+'\n')
            print(out_str)



if __name__ == '__main__':

#    file = '/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_en_train.txt'
#    model_path = '/data/caozhaojun/fasttext_model/multi_ft_en_train_weight_2e_model'
#    vec_path = '/data/caozhaojun/fasttext_model/multi_ft_en_train_cbow_300d.vec'
#    model_path = '/data/caozhaojun/fasttext_model/multi_ft_format_zh_200d_new'
#    vec_path = '/data/caozhaojun/word2vec/daodao_zh_word2vec.txt'
#    model = fasttext.cbow(file,vec_path,dim=300,word_ngrams=2,epoch=8)
#    classifier = fasttext.supervised(file,model_path,label_prefix='__label__',epoch=2,dim=300,min_count=5,pretrained_vectors=vec_path)

    lang = 'en'
    gate = 0.9
    view = True
    if lang == 'en':
 #       model_path = '/data/caozhaojun/fasttext_model/multi_ft_en_train_10e_model.bin'
        model_path = '/data/caozhaojun/fasttext_model/multi_ft_en_train_weight_2e_model.bin'
        classifier = fasttext.load_model(model_path,label_prefix='__label__',encoding='utf-8')    
        test_data = '/data/caozhaojun/true_procress_data/procressed_data_new/test_result/multi_ft_format_en_test_clean.txt'
        out_file = '/data/caozhaojun/true_procress_data/procressed_data_new/test_result/multi_ft_format_en_test_clean_predict_%.3f_new.txt' % gate
        predict_format_result_multi(test_data,out_file,gate=gate)

    elif lang == 'zh':
        model_path = '/data/caozhaojun/fasttext_model/multi_ft_format_zh_200d_new.bin'
        classifier = fasttext.load_model(model_path,label_prefix='__label__',encoding='utf-8')    
        test_data = '/data/caozhaojun/true_procress_data/procressed_data_new/test_result/multi_ft_format_zh_test_clean.txt'
        out_file = '/data/caozhaojun/true_procress_data/procressed_data_new/test_result/multi_ft_format_zh_test_clean_predict_%.3f_new.txt' % gate
        predict_format_result_multi(test_data,out_file,gate=gate)

    if view == True:
        if lang == 'en':
        #    model_path = '/data/caozhaojun/fasttext_model/multi_ft_en_train_10e_model.bin'
            model_path = '/data/caozhaojun/fasttext_model/multi_ft_en_train_weight_2e_model.bin'
            classifier = fasttext.load_model(model_path,label_prefix='__label__',encoding='utf-8')
            test_data = '/data/caozhaojun/true_procress_data/procressed_data_new/test_result/multi_ft_format_en_test_clean.txt'
            out_file = '/data/caozhaojun/true_procress_data/procressed_data_new/test_result/multi_ft_format_en_test_clean_%.3f_for_view_new.txt' % gate
            predict_format_result_multi_for_view(test_data,out_file,gate=gate)

        elif lang == 'zh':
            model_path = '/data/caozhaojun/fasttext_model/multi_ft_format_zh_200d_new.bin'
            classifier = fasttext.load_model(model_path,label_prefix='__label__',encoding='utf-8')
            test_data = '/data/caozhaojun/true_procress_data/procressed_data_new/test_result/multi_ft_format_zh_test_clean.txt'
            out_file = '/data/caozhaojun/true_procress_data/procressed_data_new/test_result/multi_ft_format_zh_test_clean_%.3f_for_view_new.txt' % gate
            predict_format_result_multi_for_view(test_data,out_file,gate=gate)

 #   model_path = '/data/caozhaojun/fasttext_model/multi_ft_format_zh_200d_new.bin'
 #   classifier = fasttext.load_model(model_path,label_prefix='__label__',encoding='utf-8')
 #   filein = "/data/caozhaojun/true_procress_data/procressed_data_new/test_result/add_zh_test_clean.txt"
 #   fileout = "/data/caozhaojun/true_procress_data/procressed_data_new/test_result/add_zh_test_clean_predict.txt"
 #   predict_format_result_multi(filein,fileout,gate=0.8) 
#    file = "/data/caozhaojun/true_procress_data/procressed_data/ft_format_data_tmp.valid"
#    out_file = "/data/caozhaojun/true_procress_data/procressed_data/valid_eval_matrix.txt"
#    do_eval(file,out_file)
#    result = classifier.test(test_data)
#    print 'labels.len',len(classifier.labels)
#    print 'model.hotel',classifier['hotel']
#    print 'P@1:', result.precision
#    print 'R@1:', result.recall
#    print 'Number of examples:', result.nexamples

#    out_file = 'ft_format_small_test_nolabel_result.txt'
#    test_data = '/data/caozhaojun/true_procress_data/test_data/small_test_data_set.txt'
#    save_result()    

#    file = '/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_zh.txt'
#    out_file = '/data/caozhaojun/true_procress_data/procressed_data_new/multi_ft_format_zh_nolabel.txt'
#    create_null_test_data(file,out_file)
