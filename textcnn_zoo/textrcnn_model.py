#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:24:44 2018

@author: miaoji
"""

import tensorflow as tf
import numpy as np
import copy
from tensorflow.python.ops import array_ops
from tflearn.data_utils import pad_sequences,to_categorical


class TextRCNN:
    def __init__(self,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,
                 vocab_size,embed_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1),multi_label_flag=True):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.hidden_size=embed_size
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.initializer=initializer
        self.activation=tf.nn.tanh
        self.multi_label_flag=multi_label_flag

        self.input_x = tf.placeholder(tf.int32, [batch_size, self.sequence_length], name="input_x")  
        self.input_y = tf.placeholder(tf.float32, [batch_size, self.num_classes], name="input_y_multilabel")  
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() 
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  
        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) 
            self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") 

        self.eval_evaluate = self.evaluate()
    
    def evaluate(self):
        '''旧方案的预测，评估模块，要删除，需要改成与textcnn一样。即textrcnn.py只输出loss与logits。预测与评估，交给textrcnn_train.py,可参照textcnn_train.py完成
        '''
        self.probs = tf.nn.softmax(self.logits)
        zeros = array_ops.zeros_like(self.probs,dtype=self.probs.dtype)
        ones = array_ops.ones_like(self.probs,dtype=self.probs.dtype)
        gate = tf.constant(0.1,shape=self.probs.shape,dtype=self.probs.dtype)
        print("gate",gate.shape)
        predictions = tf.where(self.probs > gate, ones, zeros)
        print("predictions",predictions.shape)
        
        #correct_prediction = tf.equal(tf.cast(predictions,tf.int32),tf.cast(self.input_y,tf.int32))
        #accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        return predictions

    def instantiate_weights(self):
        with tf.name_scope("weights"): 
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) 

            self.left_side_first_word= tf.get_variable("left_side_first_word",shape=[self.batch_size, self.embed_size],initializer=self.initializer) 
            self.right_side_last_word = tf.get_variable("right_side_last_word",shape=[self.batch_size, self.embed_size],initializer=self.initializer) 
            self.W_l=tf.get_variable("W_l",shape=[self.embed_size, self.embed_size],initializer=self.initializer)
            self.W_r=tf.get_variable("W_r",shape=[self.embed_size, self.embed_size],initializer=self.initializer)
            self.W_sl=tf.get_variable("W_sl",shape=[self.embed_size, self.embed_size],initializer=self.initializer)
            self.W_sr=tf.get_variable("W_sr",shape=[self.embed_size, self.embed_size],initializer=self.initializer)

            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*3, self.num_classes],initializer=self.initializer) 
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       

    def get_context_left(self,context_left,embedding_previous):

        left_c=tf.matmul(context_left,self.W_l) 
        left_e=tf.matmul(embedding_previous,self.W_sl)
        left_h=left_c+left_e
        context_left=self.activation(left_h)
        return context_left

    def get_context_right(self,context_right,embedding_afterward):
        right_c=tf.matmul(context_right,self.W_r)
        right_e=tf.matmul(embedding_afterward,self.W_sr)
        right_h=right_c+right_e
        context_right=self.activation(right_h)
        return context_right

    def conv_layer_with_recurrent_structure(self):
        embedded_words_split=tf.split(self.embedded_words,self.sequence_length,axis=1) 
        embedded_words_squeezed=[tf.squeeze(x,axis=1) for x in embedded_words_split]
        embedding_previous=self.left_side_first_word
        context_left_previous=tf.zeros((self.batch_size,self.embed_size))
        context_left_list=[]
        for i,current_embedding_word in enumerate(embedded_words_squeezed):
            context_left=self.get_context_left(context_left_previous, embedding_previous) 
            context_left_list.append(context_left) 
            embedding_previous=current_embedding_word 
            context_left_previous=context_left 
       
        embedded_words_squeezed2=copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()
        embedding_afterward=self.right_side_last_word
        context_right_afterward = tf.zeros((self.batch_size, self.embed_size))
        context_right_list=[]
        for j,current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right=self.get_context_right(context_right_afterward,embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward=current_embedding_word
            context_right_afterward=context_right
        
        output_list=[]
        for index,current_embedding_word in enumerate(embedded_words_squeezed):
            representation=tf.concat([context_left_list[index],current_embedding_word,context_right_list[index]],axis=1)
            #print(i,"representation:",representation)
            output_list.append(representation) #shape:sentence_length个[None,embed_size*3]
        
        #print("output_list:",output_list) #(3, 5, 8, 100)
        output=tf.stack(output_list,axis=1) #shape:[None,sentence_length,embed_size*3]
        #print("output:",output)
        return output


    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.max pooling, 4.FC layer 5.softmax """
        
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) #shape:[None,sentence_length,embed_size]
        
        output_conv=self.conv_layer_with_recurrent_structure() #shape:[None,sentence_length,embed_size*3]
        
        
        output_pooling=tf.reduce_max(output_conv,axis=1) #shape:[None,embed_size*3]
        
        
        with tf.name_scope("dropout"):
            h_drop=tf.nn.dropout(output_pooling,keep_prob=self.dropout_keep_prob) #[None,num_filters_total]

        with tf.name_scope("output"): 
            logits = tf.matmul(h_drop, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits

    def loss(self,l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);
           
            loss=tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def loss_multilabel(self,l2_lambda=0.00001): #0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):

            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);
            
            print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999).
            losses=tf.reduce_sum(losses,axis=1) 
            loss=tf.reduce_mean(losses)         
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op


def test():
    num_classes=3
    learning_rate=0.05
    batch_size=8
    decay_steps=1000
    decay_rate=0.9
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=0.5
    textRNN=TextRCNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training,multi_label_flag=True)
    tf.summary.scalar("loss",textRNN.loss_val)
    #tf.summary.scalar("accuracy",textRNN.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./tf_board/")
   
    print("测试textrcnn， 多标签：")
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       writer.add_graph(sess.graph)
       for i in range(2000):
           input_x=np.zeros((batch_size,sequence_length)) 
           input_x[input_x>0.5]=1
           input_x[input_x <= 0.5] = 0
           #input_y=np.array([1,0,1,1,1,2,1,1])
           input_y=np.array([[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1],[1,1,1],[1,1,1]])
           #input_y = to_categorical(input_y,3)           
           feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob}
           s = sess.run(merged_summary,feed_dict=feed_dict)
           writer.add_summary(s,i)
           
#           loss,acc,predict,W_projection_value=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.W_projection,textRNN.train_op],feed_dict)
           loss,pre,_=sess.run([textRNN.loss_val,textRNN.eval_evaluate,textRNN.train_op],feed_dict)
           if i % 400 == 0:
               print("loss:",loss,"\npredict:\n",pre,"\nlabel:\n",input_y)
               print("++++++++++++++")
       print("测试结束！")


test()
           
            




