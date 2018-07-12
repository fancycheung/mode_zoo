#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:28:40 2018

@author: miaoji
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class TextRNN:
    def __init__(self,num_classes,learning_rate, batch_size, decay_steps, decay_rate,sequence_length,
                 vocab_size,embed_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1)):
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.hidden_size=embed_size
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.initializer=initializer
        self.num_sampled=20

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  
        self.input_y = tf.placeholder(tf.int32,[None], name="input_y") #单一标签格式，此时y是单个数值，比如y=5,即y是第5个标签，下面的为向量，多标签未修改完，预测、精确度计算等参照textcnn修改。
        #self.input_y = tf.placeholder(tf.float32, [batch_size, self.num_classes], name="input_y_multilabel") #多标签格式

        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() 
        if not is_training:
            return
        #self.loss_val = self.loss() 
        self.loss_val = self.loss_nce()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) 
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") 

    def instantiate_weights(self):
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("embedding",shape=[self.vocab_size,self.embed_size],initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*2,self.num_classes],initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])
    
    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,dtype=tf.float32)
        print("outputs shape:",outputs)
        output_rnn = tf.concat(outputs,axis=2)
        self.output_rnn_last = tf.reduce_mean(output_rnn,axis=1)
        print("output_rnn_last shape:",self.output_rnn_last)
        
        with tf.name_scope("output"):
            logits = tf.matmul(self.output_rnn_last,self.W_projection) + self.b_projection
        return logits
    
    def loss(self,l2_lambda=0.001):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = l2_lambda * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) 
            loss = loss + l2_losses
        return loss
    
    def loss_nce(self,l2_lambda=0.001):
        if self.is_training:
            labels = tf.expand_dims(self.input_y,axis=1)
            
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=tf.transpose(self.W_projection),biases=self.b_projection,
                                                 labels=labels,inputs=self.output_rnn_last,num_sampled=self.num_classes,
                                                 num_classes=self.num_classes,partition_strategy="div"))
        l2_losses = l2_lambda * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) 
        loss = loss + l2_losses
        return loss
    
    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,self.decay_steps,self.decay_rate,staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val,global_step=self.global_step,learning_rate=learning_rate,optimizer="Adam")
        return train_op

def test():
    num_classes=10
    learning_rate=0.01
    batch_size=8
    decay_steps=1000
    decay_rate=0.9
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=0.5
    print("测试textrnn，单一标签：")
    textRNN=TextRNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(200):
            input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
            input_y=input_y=np.array([1,0,1,1,1,2,1,1]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss,acc,predict,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
            if i % 20 == 0:
                print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
        print("测试结束!")
test()

        
