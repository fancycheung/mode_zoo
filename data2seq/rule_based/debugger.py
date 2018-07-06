#!/usr/bin/env python
#encoding=utf-8

'''
    @desc: 打印一个行程或特征，用于调试算法
'''

from common import *
from utils import *

def route_sketch_debugger():

    print "Not Implented Yet"

def rdd_debugger():
    
    print "Not Implented Yet"

def rdf_debugger(rdf):

    for k,v in rdf.__dict__.iteritems():
        print k + " : " + str(v)

if __name__ == "__main__":

    pass

