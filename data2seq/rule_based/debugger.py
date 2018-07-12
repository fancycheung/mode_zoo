#!/usr/bin/env python
#encoding=utf-8

'''
    @desc: 打印一个行程或特征，用于调试算法
'''

from common import *
from common_class import *
from utils import *

def route_debugger(route):
    
    assert isinstance(route,Route), "fuck bug"

    info = "\n打印行程 \"%s\" 的信息，该行程共%d天\n\n"%(route.title, route.days)

    for i, (rdd, rdf, highlight, desc) in enumerate(zip(route.data_by_day, route.feature_by_day, route.highlight_by_day, route.desc_by_day)):

        info += route_day_debugger(rdd, rdf, desc, highlight) + "\n-------\n"
    
    info += '----------------------------\n'

    return info

def route_day_debugger(rdd, rdf, desc, highlight):
    
    info = "第%d天, 停留城市:%s\n行程概述: %s\n生成描述: %s"%(rdd.day_index, ', '.join(rdf.city), \
            ' -> '.join([poi.name for poi in rdf.pois]), desc)

    
    # TODO 加入行程特征信息
    # TODO 加入highlight

    return info

if __name__ == "__main__":

    pass

