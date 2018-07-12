#!/usr/bin/env python
#coding=utf-8

'''
    @desc: 定义通用数据结构
'''

# 加载基础库
from collections import defaultdict
from utils import days_delta, minutes_delta, cal_fen
from common import *
import json

# 模版数据结构定义
# 一个特征
class Feature:
    def __init__(self):
        self.name = "NULL"
        self.key = "NULL"
        self.values = []        # each ele is a string
        self.value_weight = []        # each ele is a tuple of 2 elements for (value(str), weight(int))
    
    def parse(self,line):
        items = line.strip().split('\t')
        if(len(items) != 3): return False

        self.name = items[0]
        self.key = items[1]
        fixed_weight = 50
        self.value_weight = [(item,fixed_weight) for item in items[2].split('|')]
        self.values = items[2].split('|')

        return True

# 一个词典Tag
class Word:
    def __init__(self):
        self.tag = "N"          # N: 普通词; S: 特殊词
        self.name = "NULL"
        self.root = "NULL"      # 词根
        self.values = []        # each ele is a string
        self.value_weight = []        # each ele is a tuple of 2 elements for (value(str), weight(int))

    def parse(self, line):

        items = line.strip().split('\t')
        if(len(items) != 2): return False

        self.name = items[0]
        self.root = self.name[2:][:-1]
        fixed_weight = 50
        self.value_weight = [(item,fixed_weight) for item in items[1].split('|')]
        self.values = items[1].split('|')
        
        return True

# 一个语料(POI相关)
class Corpus:
    def __init__(self):
        self.vid = "NULL"
        self.patterns = []       # 匹配语料的条件 each ele if a tuple of 2 elements for ([features], output)
                                 # each ele of features is a tuple of 2 elements (feature.name(str), feature.values(list))
    
    def add_pattern(self,line):

        items = line.strip().split('\t')
        if(len(items) != 4): return False
        features = []
        if items[2].strip() != "NULL":
            for item in items[2].strip().split('&'):
                elements = item.split(':')
                if len(elements) != 2: return False
                features.append((elements[0], elements[1].split('|')))

        output = items[3].strip()
        self.patterns.append((features,output))

        return True

# 一个输出段落(行程整体相关, entry,sentence,highlight通用结构)
class Entry:
    def __init__(self):
        self.name = "NULL"              # unique name
        self.output = "NULL"        # 标准输出，未替换变量
        self.pattern = []           # 匹配条件， each ele if a tuple of 2 elements for (feature.name, feature.values)

    def parse(self, rows):
        if(len(rows) < 3 or not rows[0].startswith("NAME::") or not rows[1].startswith("PATTERN::") or not rows[2].startswith("OUTPUT::")): return False
        self.name = rows[0].split("NAME::")[1]
        self.output = rows[2].split("OUTPUT::")[1]
        
        for item in rows[1].split("PATTERN::")[1].split('&'):
            elements = item.split(':')
            if(len(elements) != 2): continue
            self.pattern.append((elements[0], elements[1].split('|')))

        return True

# 基础数据结构定义
# 一个城市
class City:
    def __init__(self):
        self.cid = "NULL"       # 城市ID
        self.mid = "NULL"       # 所属国家ID
        self.name = "NULL"      # 城市中文名
        self.name_en = "NULL"   # 城市英文名
        self.alias = []         # 城市特色名，比如，洛杉矶: 天使之城
        self.country = "NULL"   # 所在国家
        self.view_num = -1      # 这个城市景点数量
        self.mapinfo = (0.0,0.0)   # (经度，纬度)

        self.is_park = 0        # 是否是国家公园
        self.grade = -1         # 评级


# 一个景点(也兼容广义POI)
class View:
    def __init__(self):
        # 基础信息区
        self.vid = "NULL"       # 景点ID
        self.name = "NULL"      # 景点中文名
        self.name_en = "NULL"   # 景点英文名
        self.alias = []         # 景点特色名
        self.cid = "NULL"       # 所在城市ID
        self.mapinfo = (0.0,0.0)   # (经度，纬度) 

        self.intro = "NULL"
        
        # 特殊标记区
        self.type_tag = "N"   # N: 普通景点; C: 市中心(用于指代自由活动); A: 机场; S: 火车站; H: 酒店;
                              # 除N外，都是虚拟点, S也指代机场外的公共交通站点

        # 景点特色区
        self.tags = []      # 景点特色, 建筑人文、历史遗迹等 , 参考tag表
        self.rank = -1          # 景点排名
        self.rank_region = 10  # 排名在城市中 top x%, 1-100整数
        self.play_rcmd = ("深度", 60)   # 推荐的游玩时长，第一个元素是游览方式，第二个是推荐时长(分)
        
# 行程数据结构相关
# [!]这里只是定义，解析程序在route_parser/feature_parser中实现
# 区别：1. 数据从原始Json解析，赋值到各个成员变量；特征从对应的数据类解析
#       2. 数据多是数值型的，特征多是描述字符串。比如出发时间，在数据中是具体的时间09:00，在特征中可能是早上
#       3. 数据值由行程数据确定，特征值取决于数据值&控制策略，如09:00 可能是早上，也可能是上午，取决于模型策略
# 一个POI点 数据
class POIData:
    def __init__(self):
        self.vid = "NULL"       # POI ID
        self.start_time = 0     # 开始时间
        self.end_time = 0       # 结束时间
        self.dur = 0            # 持续时间
        self.do_what = "NULL"   # 干啥
        self.play = "NULL"      # 咋玩
        self.dining_nearby = 0  # 是否在附近就餐, 0 No; 1 Yes
        self.traffic = (0,0,0)  # 离开此点到下一个点的交通，分别是type,distance(meter),time(minute)

# 一个POI点 特征
class POIFeature:
    def __init__(self):
        self.vid = "NULL"       # poi ID
        self.name = "NULL"      # poi Name
        self.cid = "NULL"       # city ID
        self.city = "NULL"      # city Name
        self.next_city = "NULL" # next_city Name
        self.tag = "U"          # undefined, C,S,A,H,N etc
        self.start_time = "NULL"     # 开始时间
        self.end_time = "NULL"       # 结束时间
        self.dur = "NULL"            # 持续时间
        self.play = "NULL"      # 游玩方式
        self.dining_nearby = "N"    # Y/N
        self.traffic = ("NULL","NULL","NULL")
        self.corpus = "NULL"
        
        self.is_first_poi = "N"     # 该天该城市的第一个点
        self.is_last_poi = "N"      # 该天该城市的最后一个点
        self.is_first_view = "N"    # 该天该城市的第一个游玩点
        self.is_last_view = "N"     # 该天该城市的最后一个游玩点
        self.is_leaving_city = "N"  # 是否即将离开该城市，只有离开交通才能是Y


# 一天的行程 数据
class RouteDayData:
    def __init__(self):
        self.day_index = -1     # 整个行程的第几天
        self.on_the_way = 0     # 是否整天都在交通工具上
        self.date = "19000101"
        
        # 以下结构的长度都相等，都等于len(self.city)
        self.city = []                  # 这一天所在的城市cid
        self.time_by_city = []          # 归属于每个城市内的(stime,etime,dur)
        self.city_day_index = []        # 分别是对应城市的第几天
        self.city_index = []            # 分别是第几个城市
        self.city_traffic_hotel = []    # 是否需要traffic和hotel
        self.views = []                 # list of list， 每个城市安排的景点
        
# 一天的行程 特征
# [!]解析程序在route_parser中实现
class RouteDayFeature:
    def __init__(self):

        self.city_num = "NULL"      # 城市数量的描述
        self.city = []              # 城市name的列表
        self.play_city_str = "NULL" # 所有游玩城市串起来
        self.play_first_day = "N"   # 整个行程的第一天游玩
        self.play_last_day = "N"    # 整个行程的最后一天游玩

        self.date = "19000101"
        self.season = "NULL"        # 所在季节
        self.month = "NULL"         # 所在月份

        # 行程描述部分特征
        self.pois = []              # poi点安排, 每个元素是一个PoiFeature
        self.poi_sentences = []     # 每个poi点的描述
        
        # 行程亮点部分特征 TODO
    '''
        self.play_city_num = 0      # 游玩城市个数 [!]不等同于停留城市个数
        self.single_city = "U"      # Y: yes; N: no; U: unknown
    
        self.trip_first_day = "N"   # 整个行程的第一天
        self.trip_last_day = "N"    # 整个行程的最后一天

        # for single_city = "Y"
        self.city = "NULL"          # cid
        self.is_park = "N"          # 是否是国家公园
        self.start_time = "NULL"    # 开始游玩时刻，早上、中午、晚上
        self.end_time = "NULL"      # 结束游玩时刻，早上、中午、晚上
        self.start_poi = "NULL"     # 城市内出发地点: 机场、车站或酒店
        self.end_poi = "NULL"       # 城市内结束地点: 机场、车站或酒店
        self.city_first_day = "U"   # 该城市的第一天
        self.city_last_day = "U"    # 该城市最后一天
        self.hot_level = "NULL"     # 游玩景点热门程度
        self.dur = "NULL"           # 游玩时长特征值
        self.play = "NULL"          # 这一天的游览方式：深度、匆匆等
        self.view_num = "NULL"      # 游玩景点数量多少
        self.view_distance = "NULL" # 景点间距离远近
        self.single_main_view = "N" # 平行景点还是有主要景点
        self.main_view = "NULL"     # 主要游玩景点
        self.view_locate = "NULL"   # 景点位置  主要考虑距旅馆的位置
        self.feature = "NULL"       # 景点安排的主要特色
    '''

# 一个行程
class Route:
    def __init__(self):
        
        # 基本信息
        self.days = 0               # 整个行程的天数
        self.city_num = 0           # 目的地数量(包括首、尾)
        self.dept_city = "NULL"     # 出发城市cid
        self.back_city = "NULL"     # 返回城市cid
        self.dept_date = "19000101" # 行程出发日期
        self.back_date = "20991231" # 行程结束日期
        self.dept_time = 0          # 整个行程出发时刻，分钟数
        self.back_time = 24*60 - 1  # 整个行程结束时刻，分钟数
        self.title = "NULL"         # 行程标题
        
        # 按天分割
        self.data_by_day = []     # each ele should be an instance of RouteDayData
        self.feature_by_day = []  # each ele should be an instance of RouteDayFeature
        
        # 存储生成的结果
        self.highlight_by_day = []  # one highlight each day, string is ok
        self.desc_by_day = []       # one description each day, string is ok

if __name__ == "__main__":

    #test 
    pass
