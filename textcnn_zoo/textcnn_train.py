#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:26:28 2018

@author: miaoji
"""
import tensorflow as tf
import numpy as np
from textcnn_model import TextCNN
from tflearn.data_utils import to_categorical, pad_sequences
import os
import pickle
from tc_utils import create_voabulary,create_voabulary_label,load_data,load_word2vec,softmax,evaluate,batch_iter
import time
import os

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",16,"number of label") #目前标签数为16
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size for training/evaluating.") 
tf.app.flags.DEFINE_integer("decay_steps", 100, "how many steps before decay learning rate.") 
tf.app.flags.DEFINE_float("decay_rate", 0.65, "Rate of decay for learning rate.") 
tf.app.flags.DEFINE_integer("sentence_len",200,"max sentence length") #句子长度设置为200
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",20,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every",1, "Validate every validate_every epochs.") 
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters",200, "number of filters") #卷积核数200个
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
filter_sizes = [1,2,3,4,5,6,7,8,9,10] 
_LANG = "en"
tf.app.flags.DEFINE_string("ckpt_dir","textcnn_checkpoint_%s/" % _LANG,"checkpoint location for the model")
if _LANG == "en": # 分别导入中、英文词向量
    tf.app.flags.DEFINE_string("word2vec_model_path","./cache_pickle//multi_ft_en_train_cbow_300d.vec","word2vec") 
    tf.app.flags.DEFINE_integer("embed_size",300,"embedding size") #英文词向量为300维
elif _LANG == "zh":
    tf.app.flags.DEFINE_string("word2vec_model_path","./cache_pickle//daodao_zh_word2vec.txt","word2vec")
    tf.app.flags.DEFINE_integer("embed_size",200,"embedding size") #中文词向量为200维

def main(_):
    time_start = time.time()
    
    def predict_label(logits,tmp_gate=0.01):#根据logit，预测label，预测为大于设定的阈值得标签
        probs = softmax(logits)

        labels = []
        for prob in probs:
            con = np.greater_equal(prob,[tmp_gate]*16) #判断是否大于阈值，此处为0.01
            tmp = list(np.argwhere(con == True))
            label = [x[0] for x in tmp]
            if sum(label) < 1:#如果没有大于阈值的label，就选最大的label
                label = np.argmax(prob)
            labels.append(label)
            #in_prob = prob[index]
        return labels
   
    def predict_label_top_k(sess,eval_return):
        '''旧方案预测top k标签，k可以固定下来或者根据修改model.py，动态确定每条评论的k'''
        y_predict = [] 
        top_number = eval_return[0] # 每条评论的标签数
        probs_squeezed = eval_return[1] # logits
        for i,curr_prob in enumerate(probs_squeezed):
            index = tf.nn.top_k(curr_prob,top_number[i]) #从logits中选取对应前k个
            index = set(index.indices.eval())
            y_predict.append(tf.constant([1 if i in index else 0 for i in range(FLAGS.num_classes)]))
        y_predict = tf.stack(y_predict)
        return y_predict.eval()

    def pad_y(Y,label_num=16):#根据标签在标签集的整数索引，转化为multi-hot向量
        Y2 = np.zeros((len(Y),label_num))
        for i,y in enumerate(Y):
            tmp_y = [0]*label_num
            for it_y in y:
                tmp_y[it_y] = 1
            Y2[i,:] = tmp_y
        return Y2

    vocabulary_word2index, vocabulary_index2word = create_voabulary(file=FLAGS.word2vec_model_path,cache_path = "./cache_pickle/ft_%s_voabulary.pickle" % _LANG,from_word2vec=1) # 导入词汇表
    vocab_size = len(vocabulary_word2index) #词汇量

    trainX, trainY, testX, testY = None, None, None, None

    cache_train_data_path = "./cache_pickle/train_data_%s.pickle" % _LANG #导入由tc_utils.py模块准备好的训练集、验证集
    train_data = os.path.exists(cache_train_data_path)
    cache_eval_data_path = "./cache_pickle/eval_data_%s.pickle" % _LANG
    eval_data = os.path.exists(cache_eval_data_path)
    if (train_data and eval_data):
        with open(cache_train_data_path,'rb') as f:
            trainX,trainY = pickle.load(f)
        with open(cache_eval_data_path,'rb') as f:
            testX,testY = pickle.load(f)
    else:
        return "data NOT found!"

    trainY = pad_y(trainY)
    testY = pad_y(testY)

    trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.) #对评论进行截断或者补全到固定长度 
    testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)    
    
    config=tf.ConfigProto()
    best_f1 = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 500  # 如果超过500轮未提升，提前结束训练
    print_per_batch = 10 # 每多少轮次输出在训练集和验证集上的性能
    save_per_batch = 50 # 每多少轮次将训练结果写入tensorboard scalar
    save_path=FLAGS.ckpt_dir + "model.ckpt"
    total_batch = 1
    flag = False

    with tf.Session(config=config) as sess:

        print("initialize model")
        textCNN=TextCNN(filter_sizes,FLAGS.num_filters,FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training,multi_label_flag=FLAGS.multi_label_flag) #初始化模型，可替换为textrcnn、textrnn
       
        tf.summary.scalar("loss",textCNN.loss_val)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./tf_board/")

        saver=tf.train.Saver() 
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):#可导入以前训练好的模型，继续进行训练
            print("Restore Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Initialize variable")
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #使用训练好的词向量
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN,cache_path="./cache_pickle/embedding_%s.pickle" % _LANG,word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch=sess.run(textCNN.epoch_step)

        writer.add_graph(sess.graph)

        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        print("Start epoch") #开始训练模型
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            
            batch_train = batch_iter(trainX,trainY,batch_size)
            for curr_trainX,curr_trainY in batch_train:

                feed_dict = {textCNN.input_x: curr_trainX,textCNN.dropout_keep_prob: 0.5}
                feed_dict[textCNN.input_y] = curr_trainY
                
                if total_batch == 1:
                    print('testX\n',testX)
                    print('testY\n',testY)

                if total_batch % save_per_batch == 0:
                    s = sess.run(merged_summary,feed_dict=feed_dict)
                    writer.add_summary(s,total_batch)
                    
                if total_batch % print_per_batch == 0:
                    feed_dict[textCNN.dropout_keep_prob] = 1.0
                    #train_loss,train_logits = sess.run([textCNN.loss_val,textCNN.logits],feed_dict=feed_dict)
                    feed_dict1 = {textCNN.input_x:testX,textCNN.input_y:testY,textCNN.dropout_keep_prob:1.0}
                    test_loss,logits = sess.run([textCNN.loss_val,textCNN.logits],feed_dict=feed_dict1)
                    predict_y = predict_label(logits)
#                    predict_y = predict_label(test_logits)
                    test_acc,precision,recall,f1 = evaluate(predict_y,testY) # waitting
                
                    if test_acc > best_f1:
                        best_f1 = test_acc
                        last_improved = total_batch
                        saver.save(sess,save_path,global_step=total_batch)
                        improved_str = '*'
                    else:
                        improved_str = ''
                    print("epoch:%d total_batch:%d test_loss:%f test_acc:%f precision:%f recall:%f f1:%f %s" % (epoch,total_batch,test_loss,test_acc,precision,recall,f1,improved_str))  #waitting
                sess.run([textCNN.train_op],feed_dict) 
                 
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    print("auto stopping")
                    flag = True
                    break
            if flag:
                break
            
    time_end = time.time()
    print('using time:',time_end-time_start)

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textCNN,cache_path = "",word2vec_model_path=None):
    cache_path_exist = os.path.exists(cache_path)
    if cache_path_exist == True:
        with open(cache_path, 'rb') as data_f:
            word_embedding_final = pickle.load(data_f)       
    else: 
        print("using pre-trained word emebedding:",word2vec_model_path)
        word2vec_dict = load_word2vec(word2vec_model_path,cache_path = "./cache_pickle/ft_train_word2vec_%s.pickle" % _LANG)
        word_embedding_2dlist = [[]] * vocab_size
        print('create end')
        word_embedding_2dlist[0] = list(np.zeros(FLAGS.embed_size))
        bound = np.sqrt(6.0) / np.sqrt(vocab_size)
        count_exist = 0
        count_not_exist = 0
        for i in range(1, vocab_size):
            word = vocabulary_index2word[i]
            embedding = None
            try:
                embedding = word2vec_dict[word]
            except Exception:
                embedding = None
            if embedding is not None:
                word_embedding_2dlist[i] = embedding;
                count_exist = count_exist + 1
            else:
                word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
                count_not_exist = count_not_exist + 1
        word_embedding_final = np.array(word_embedding_2dlist)
        print("word exists embedding:", count_exist, "word not exist embedding:", count_not_exist)
        with open(cache_path,'ab') as data_f:
            pickle.dump(word_embedding_final,data_f)
    print('word_embedding_final.shape',word_embedding_final.shape)
    print(word_embedding_final) 
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)
    t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)
    sess.run(t_assign_embedding);


if __name__ == "__main__":
    tf.app.run()



