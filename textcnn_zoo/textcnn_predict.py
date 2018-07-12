#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:23:03 2018

@author: miaoji
"""

import tensorflow as tf
import numpy as np
from textcnn_model import TextCNN
from tflearn.data_utils import to_categorical, pad_sequences
import os
import pickle
from tc_utils import create_voabulary,create_voab_label,load_data,load_word2vec,softmax,evaluate,batch_iter
import time

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",16,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.") 
tf.app.flags.DEFINE_integer("decay_steps", 100, "how many steps before decay learning rate.") 
tf.app.flags.DEFINE_float("decay_rate", 0.65, "Rate of decay for learning rate.") 
tf.app.flags.DEFINE_integer("sentence_len",200,"max sentence length")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every",1, "Validate every validate_every epochs.") 
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters",200, "number of filters") 
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
filter_sizes = [1,2,3,4,5,6,7,8,9,10]

_LANG = "en"  # "zh en"中、英文语言

tf.app.flags.DEFINE_string("ckpt_dir","textcnn_checkpoint_%s/" % _LANG,"checkpoint location for the model")
tf.app.flags.DEFINE_string("traning_data_path","./cache_pickle/ft_%s.eval" % _LANG,"path of test data.") 
if _LANG == "en":
    tf.app.flags.DEFINE_string("word2vec_model_path","./cache_pickle/multi_ft_en_train_cbow_300d.vec","word2vec")
    tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
elif _LANG == "zh":
    tf.app.flags.DEFINE_string("word2vec_model_path","./cache_pickle/daodao_zh_word2vec.txt","word2vec")
    tf.app.flags.DEFINE_integer("embed_size",200,"embedding size")

def main(_):

    time_start = time.time()

    def save_predict(predict_y,voc,file):
        print(predict_y)
        with open(file,'w') as f:
            for i in range(len(predict_y)):
                if isinstance(predict_y[i],list):
                    labels = []
                    for j in predict_y[i]:
                        label = voc[j]
                        labels.append(label)
                        line = ["__label__"+x for x in labels]
                        if len(predict_y[i]) > 1 and "__label__NULL" in line:
                            line.remove("__label__NULL")
                        line = " ".join(line)
                else:
                    label = voc[predict_y[i]]
                    line = "__label__"+label
                f.write(line+"\n")

    def predict_label(logits):
        probs = softmax(logits)
        labels = []
        for prob in probs:
            con = np.greater_equal(prob,[0.01]*16)
            tmp = list(np.argwhere(con == True))
            label = [x[0] for x in tmp]
            if sum(label) < 1:
                label = np.argmax(prob)
            labels.append(label)
        return labels

    def predict_label_top_k(sess,eval_return,batch_size=1):
        top_number = eval_return[0]
        probs = eval_return[1]
        ones = tf.ones(shape=top_number.shape,dtype=tf.float32)
        top_number = tf.cast(tf.where(tf.greater(top_number,ones),top_number,ones),dtype=tf.int32)

        probs_split = tf.split(probs,batch_size)
        probs_squeezed = [tf.squeeze(x) for x in probs_split]

        y_predict = []
        for i,curr_prob in enumerate(probs_squeezed):
            index = tf.nn.top_k(curr_prob,top_number[i])
            index = set(index.indices.eval())
            y_predict.append(tf.constant([1 if i in index else 0 for i in range(FLAGS.num_classes)]))
        y_predict = tf.stack(y_predict)
        return y_predict.eval()

    def pad_y(Y):
        Y2 = np.zeros((len(Y),16))
        for i,y in enumerate(Y):
            tmp_y = [0]*16
            for it_y in y:
                tmp_y[it_y] = 1
            Y2[i,:] = tmp_y
        return Y2

    testX, testY = None, None
    
    cache_test_data_path = "./cache_pickle/eval_data_%s.pickle" % _LANG
    test_data = os.path.exists(cache_test_data_path)
    vocabulary_word2index_label,vocabulary_index2word_label = create_voab_label()  
    vocabulary_word2index, vocabulary_index2word = create_voabulary(file=FLAGS.word2vec_model_path,cache_path = "./cache_pickle/ft_%s_voabulary.pickle" % _LANG,from_word2vec=1)

    if not test_data:
        print("test data NOT exist")
        vocab_size = len(vocabulary_word2index) 
        print("cnn_model_vocab_size:",vocab_size)

        testX,testY = load_data(vocabulary_word2index, vocabulary_word2index_label,training_data_path=FLAGS.traning_data_path,cache_path='')
        print("testX:",len(testX),"testY:",len(testY))

        testY = pad_y(testY)
        testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)
            
        with open(cache_test_data_path, 'ab') as data_f:
            pickle.dump((np.array(testX),np.array(testY)),data_f)
        print("dump data end!")

    else:
        vocab_size = len(vocabulary_word2index)
        print("cnn_model_vocab_size:",vocab_size)
        with open(cache_test_data_path, 'rb') as data_f:
            testX,testY = pickle.load(data_f)
    testY = pad_y(testY)
    testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)

    config=tf.ConfigProto()
    with tf.Session(config=config) as sess:
        print("initialize model")
        textCNN=TextCNN(filter_sizes,FLAGS.num_filters,FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training,multi_label_flag=FLAGS.multi_label_flag)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint. going to stop")
            return
        feed_dict = {textCNN.input_x:testX,textCNN.input_y:testY,textCNN.dropout_keep_prob:1.0}
        test_loss,logits = sess.run([textCNN.loss_val,textCNN.logits],feed_dict=feed_dict)
        # 这里是验证，所以有loss，如果只预测，就不用loss，feed_dict也不用textCNN.input_y
        # feed_dict = {textCNN.input_x:testX,textCNN.dropout_keep_prob:1.0}
        # logits = sess.run([textCNN.logits],feed_dict=feed_dict)
        predict_y = predict_label(logits)
        save_predict(predict_y,file='./result_%s.txt' % _LANG,voc=vocabulary_index2word_label)
        
        test_acc,precision,recall,f1 = evaluate(predict_y,testY)
        print("test_loss:%f test_acc:%f precision:%f recall:%f f1:%f" % (test_loss,test_acc,precision,recall,f1))

if __name__ == "__main__":
    tf.app.run()
         



