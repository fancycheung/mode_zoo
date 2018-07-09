#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:44:30 2018

@author: miaoji
"""
import tensorflow as tf
import numpy as np
from tflearn.data_utils import pad_sequences,to_categorical
from tensorflow.python.ops import array_ops
from tc_utils import evaluate,softmax
import pickle
import math

class TextCNN:
    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1),multi_label_flag=False,clip_gradients=5.0,decay_rate_big=0.50,gate=0.15):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.is_training=is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.initializer=initializer
        self.num_filters_total=self.num_filters * len(filter_sizes)
        self.multi_label_flag=multi_label_flag
        self.clip_gradients = clip_gradients
        self.gate = gate
#        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
       # self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name="input_x")
       # self.input_y = tf.placeholder(tf.float32,[self.batch_size,self.num_classes], name="input_y_multilabel")
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32,[None,self.num_classes], name="input_y_multilabel")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

#        self.length_y = tf.cast(tf.count_nonzero(self.input_y,1),dtype=tf.float32)  #### 
        self.instantiate_weights()
        self.logits = self.inference()  #### 

#        if not is_training:
#            return

#        self.loss_val = self.focal_loss()
        self.loss_val = self.loss_multilabel()
        self.eval_evaluate = self.evaluate()
        if not is_training:
            return

        self.train_op = self.train()
        
    
    def evaluate(self): ####
        probs = tf.nn.softmax(self.logits)
        return probs


    def get_position_encoding(self,inputs, hidden_size=300, min_timescale=1.0, max_timescale=1.0e4):
        length = tf.shape(inputs)[1]
        position = tf.to_float(tf.range(length))
        num_timescales = hidden_size // 2
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
                tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        return signal
     
    def instantiate_weights(self):
        with tf.name_scope("embedding"): 
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) 
            self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total, self.num_classes],initializer=self.initializer) 
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])

    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        self.position_embedding = self.get_position_encoding(self.embedded_words,self.embed_size)  ####
        self.embedded_words = self.embedded_words + self.position_embedding  ####
        self.sentence_embeddings_expanded=tf.expand_dims(self.embedded_words,-1)

        pooled_outputs = []
        with tf.name_scope("convolution-pooling"):
            for i,filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("convolution-pooling-%s" %filter_size):
                    filter=tf.get_variable("filter-%s"%filter_size,[filter_size,self.embed_size,1,self.num_filters],initializer=self.initializer)
                    conv=tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1,1,1,1], padding="VALID",name="conv")
                    b=tf.get_variable("b-%s"%filter_size,[self.num_filters])
                    h=tf.nn.relu(tf.nn.bias_add(conv,b),"relu")
                    pooled=tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")
                    pooled_outputs.append(pooled)
                
            self.h_pool=tf.concat(pooled_outputs,3)
            self.h_pool_flat=tf.reshape(self.h_pool,[-1,self.num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob) 

        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop,self.W_projection) + self.b_projection  
        return logits 


    def focal_loss(self,weights=None, alpha=0.5, gamma=2,l2_lambda=0.00001):
        with tf.name_scope("focal_loss"):
            prediction_tensor = self.logits
            target_tensor = self.input_y
            sigmoid_p = tf.nn.sigmoid(prediction_tensor)
            zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
#       pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
            pos_p_sub = tf.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
#       neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
            neg_p_sub = tf.where(target_tensor > zeros, zeros, sigmoid_p)
        
            per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                                  - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        
            losses = tf.reduce_sum(per_entry_cross_ent,axis=1)
            loss=tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
            
        return loss 

    def loss_multilabel(self,l2_lambda=0.0001,task2_lambda=0.000001): 
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
#            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:",losses) 
            losses=tf.reduce_sum(losses,axis=1) 
            loss=tf.reduce_mean(losses)         
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        with tf.name_scope("train"):
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
            train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op



def test():
    num_classes=193
    learning_rate=0.001
    batch_size=100
    decay_steps=1000
    decay_rate=0.9
    sequence_length=100
    vocab_size=317729
    embed_size=300
    is_training=True
    dropout_keep_prob=0.5
    filter_sizes=[3,5,7,9]
    num_filters= 20
    textRNN=TextCNN(filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training)
    
    tf.summary.scalar("loss",textRNN.loss_val)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./tf_board/")
    
    cache_train_data_path = "./cache_pickle/raw_test.pickle"

    with open(cache_train_data_path, 'rb') as data_f:
        trainX,trainY = pickle.load(data_f)

    trainY2 = []
    for y in trainY:
        tmp_y = [0]*193
        for it_y in y:
            tmp_y[it_y] = 1
        trainY2.append(np.asarray(tmp_y))
    trainY = np.asarray(trainY2)
    trainX2 = []
    for x in trainX:
        trainX2.append(np.asarray(x))
    trainX = np.asarray(trainX2)
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    print("end pad!")
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       writer.add_graph(sess.graph)
       start = 0
       print("start run")
       for i in range(100):
           end = start + batch_size
           input_x = trainX[start:end]
           input_y = trainY[start:end]
           start = end
           feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob}
           s = sess.run(merged_summary,feed_dict=feed_dict)
           writer.add_summary(s,i)
           
#           loss,acc,predict,W_projection_value=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.W_projection,textRNN.train_op],feed_dict)
           #loss =sess.run([textRNN.loss_val],feed_dict)
           print("sess run",i)
           loss,_ = sess.run([textRNN.loss_val,textRNN.train_op],feed_dict)
#               a,p,r = evaluate(pre,input_y)
#               print(input_y)
           print("loss:",loss)#,'a',a,'p',p,'r',r)
#           print("W_projection_value_:",W_projection_value)
           
if __name__ == "__main__":
    test()






