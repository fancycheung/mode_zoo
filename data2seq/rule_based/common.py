#!/usr/bin/env python
#coding=utf-8

'''
    @desc: 一些全局变量，策略性的通用函数定义
'''

# 加载基础库
from collections import defaultdict
from utils import days_delta, minutes_delta, cal_fen
import datetime
import math
import json
import re
import random

# 一些通用变量
DATE_FORMAT = "%Y%m%d"
TIME_FORMAT = "%Y%m%d_%H:%M"
 
# 一个行程Feature
BASEDATA_IP = "10.10.99.230"
BASEDATA_DB = "base_data_test"

# 模板实体对应
ENTITY_INDEX = {"route":0, "day":1, "poi":2}

# 虚拟VID
VID_CITY, VID_HOTEL, VID_AIRPORT, VID_STATION, VID_VIEW, VID_VIEW_UTIL = \
        "v_city_center","v_hotel","v_airport","v_station","v_view","v_view_util"
VIRTUAL_VIEWS = [VID_CITY, VID_HOTEL, VID_AIRPORT, VID_STATION, VID_VIEW, VID_VIEW_UTIL]

# 提取特殊标记
TAGPAT = re.compile(r'(\$\[(?:.+?)\])',re.S)
TAGSPLITER = "->"
INDEXPAT = re.compile(r'(\((?:.+?)\))',re.S)
ALTERPAT = re.compile(r'(\[(?:.+?)\])',re.S)

# 特殊词列表
SPECIAL_WORDS = ["$[city]", "$[date]", "$[play_city_num]", "$[main_view]"]

# 特征解析相关定义
# [数字转成汉字]
NUM_TO_ZI = {1:"一",2:"二",3:"三",4:"四",5:"五",6:"六",7:"七",8:"八",9:"九","10":"十"}

# [通过日期判定季节]0301-0530,春季; 0601-0830,夏季; 0901-1131,秋季;1201-0229,冬季
SEASON_DATE = {"winter":(1200,230),"autumn":(900,1132),"summer":(600,832),"spring":(300,532)}

# [通过时间点判断时间区间]7-, 凌晨; 7-9.5点, 早上; 9.5-12点, 上午; 12-16点, 下午; 16-18点,黄昏; 18+,夜里
TIME_RANGE = {"early":(-1,420), "morning":(421,570), "forenoon":(571,720), \
        "afternoon":(721,960), "nightfall":(961,1080), "evening":(1081,50000)}

# [景点数量级别]0:no; 1:single; 2-3:seldom; 4-5:some; 6+:many
VIEW_NUM_RANGE = {"no":(0,0), "single":(1,1), "seldom":(2,3), "some":(4,5), "many":(6,1000)}

# [景点热度级别]>90: hottest; >80: hot; else ok;
HOT_LEVEL = {"hottest":(90,101), "hot":(80,89), "ok":(0,79)}

# [游玩时长级别]>8h: allday; >6h: mostday: else halfday
ALL_DAY_MINUTE = 600
DUR_RANGE = {"allday":(0.8,100), "mostday":(0.6,0.8), "halfday":(0.0,0.6)}

# [判断时长是否可以自由活动]
FREE_VIEW_THRESHOLD = 150       # 2.5h

# 通用数据相关函数
def get_norm_vid(typ, vid):
    # 归一化ID, 参考https://www.tapd.cn/21405211/markdown_wikis/#1121405211001003431

    if typ == 2:
        return vid
    elif typ == 16:
        return VID_AIRPORT
    elif typ == 1:
        return VID_CITY
    elif typ == 4:
        return VID_HOTEL
    elif typ == 32 or typ == 64 or typ == 128:
        return VID_STATION

    return VID_CITY

def get_norm_trans_way(way, baoche=0):
    # 归一化交通方式,0其他 1驾车 2步行 3公交 4地铁 5火车 6有轨电车

    if way == 3 or way == 4 or way == 5 or way == 6:
        return "public"
    elif way == 1:
        if baoche == 0:
            return "drive"
        elif baoche == 1:
            return "baoche"
    elif way == 2:
        return "walk"
    
    return "NULL"

# 特征值计算相关函数
def get_season(date):

    day = int(date[4:])
    for k,v in SEASON_DATE.iteritems():
        if day > v[0] and day <= v[1]:
            return k

    return "unknown"

def get_month(date):

    return str(int(date[4:6]))

def get_time_range(fen):

    for k,v in TIME_RANGE.iteritems():
        if fen > v[0] and fen <= v[1]:
            return k

    return "NULL"

def get_view_num(x):
    
    for k,v in VIEW_NUM_RANGE.iteritems():
        if x >= v[0] and x <= v[1]:
            return k

    return "no"

def get_poi_tag(vid):
    tag = "view"

    if vid == VID_CITY: tag = "city"
    elif vid == VID_HOTEL: tag = "hotel"
    elif vid == VID_AIRPORT: tag = "airport"
    elif vid == VID_STATION: tag = "station"

    return tag

def get_play_city_num(rdd, view_dict):

    # 判断在几个城市内游玩
    # 这一天可能在多个城市，但一个城市玩，另一个城市只是刚到达（离开）
    # 或这一天在一个城市，但主要时间都是交通

    # FIXME 这里先给一个固定值

    return 1

def get_hot_level(rdd, view_dict):

    # 计算这一天景点的平均热度，百分制 比如top3%的景点，给97分
    score, num = 0, 0
    for v in rdd.views:
        if view_dict.has_key(v) and view_dict[v].type_tag == "N":
            score += 100 - view_dict[v].rank_region
            num += 1

    if num > 0:
        score = score / num
        for k,v in HOT_LEVEL.iteritems():
            if score >= v[0] and score <= v[1]:
                return k
    
    return "ok"

def get_dur_level(rdd, view_dict):

    # 计算游玩时间占全天活动时间的比例, 一天按10小时计算
    ratio = float(rdd.dur) / ALL_DAY_MINUTE
    for k,v in DUR_RANGE.iteritems():
        if ratio >= v[0] and ratio <= v[1]:
            return k

    return "halfday"

def get_dist_desc(dist):
    desc = "一会"
    if dist < 100:
        desc = "不到100米"
    elif dist < 1000:
        hundred = dist / 100
        desc = "约" + NUM_TO_ZI[hundred] + "百米"
    elif dist >= 1000 and dist <= 2000:
        desc = "一千多米"
    elif dist < 10000:
        thousand = dist / 1000
        desc = NUM_TO_ZI[thousand] + "千米"
    else:
        desc = "十几千米"

    return desc

def get_dur_desc(dur):
    
    if dur <= 30:
        desc = "一会"
    elif dur <= 60:
        desc = "半个多小时"
    elif dur >= 540:
        desc = "一整天"
    else:
        hour = dur / 60
        minute = dur % 60
        if minute < 30:
            desc = "约" + NUM_TO_ZI[hour] + "个半小时"
        else:
            desc = "约" + NUM_TO_ZI[hour+1] + "个小时"
    
    return desc

def get_moment_by_time(time):
    
    # FIXME 优化特征
    return get_time_range(time)
    
if __name__ == "__main__":

    #test 
    test_tag = "$[city]$[hotviews]一网打尽"
    print TAGPAT.findall(test_tag)
