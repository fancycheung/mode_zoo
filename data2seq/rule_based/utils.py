#!/usr/bin/env python
#coding=utf-8

'''
    @desc: 通用函数或常量定义，与具体数据结构无关
'''

import math
from math import pi, radians, sin, cos
import datetime

EARTH_RADIUS = 6378137

def rad(d):
    return d * pi / 180.0

# 经纬度距离计算公式，都按[米]计算，输入参数是经度1，纬度1，经度2，纬度2
def haversine(lng1, lat1, lng2, lat2):

    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)

    s = 2 * math.asin(math.sqrt(math.pow(sin(a/2), 2) + cos(radLat1) * cos(radLat2) *math.pow(sin(b/2), 2)))
    s = s * EARTH_RADIUS

    return s

def days_delta(d1, d2, date_format="%Y%m%d"):
    # d2要比d1大
    return (datetime.datetime.strptime(d2,date_format) - datetime.datetime.strptime(d1,date_format)).days

def minutes_delta(t1, t2, time_format="%Y%m%d_%H:%M"):

    return (datetime.datetime.strptime(t2,time_format) - datetime.datetime.strptime(t1,time_format)).total_seconds() / 60

def cal_fen(timeStr):

    return int(timeStr.split(':')[0]) * 60 + int(timeStr.split(':')[1])

if __name__ == "__main__":

    v1 = "18.072319,59.3254509"
    v2 = "2.324095,48.86943"
    
    lng1,lat1 = float(v1.split(',')[0]), float(v1.split(',')[1])
    lng2,lat2 = float(v2.split(',')[0]), float(v2.split(',')[1])
    print haversine(lng1, lat1, lng2, lat2)
