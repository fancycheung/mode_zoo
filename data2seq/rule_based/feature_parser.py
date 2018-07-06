#!/usr/bin/env python
#coding=utf-8

'''
    @desc: 解析行程特征
'''

from common import *
from common_class import *

# 提取一天行程的特征
def day_feature_parser(route, rdd, rdf, city_dict, view_dict):
    
    assert isinstance(route, Route), "route is not an instance of Route"
    assert isinstance(rdd, RouteDayData), "rdd is not an instance of RouteDayData"
    assert isinstance(rdf, RouteDayFeature), "rdf is not an instance of RouteDayFeature"
    
    rdf.date = rdd.date
    rdf.season = get_season(rdd.date)
    rdf.month = get_month(rdd.date)
    
    if len(rdd.city) == 1:
        rdf.city_num = "one_city"
    elif len(rdd.city) == 2:
        rdf.city_num = "two_city"
    else:
        rdf.city_num = "many_city"
    
    rdf.city = [city_dict[cid].name for cid in rdd.city]

    if len(rdd.city_index) > 1 and rdd.city_index[0] == 0:
        rdf.play_first_day = "Y"
    if len(rdd.city_index) > 1 and rdd.city_index[-1] == route.city_num-1:
        rdf.play_last_day = "Y"
    
    play_city = []
    for i in range(len(rdd.city_index)):
        index = rdd.city_index[i]
        if index ==0 or index == route.city_num-1: continue
        play_city.append(city_dict[rdd.city[i]].name)
    if len(play_city) > 1:
        rdf.play_city_str = "，".join(play_city[:-1]) + "和" + play_city[-1]
    elif len(play_city) == 1:
        rdf.play_city_str = play_city[0]
    else:
        rdf.play_city_str = ""

    def extract_poi_feature(pf,index):
        # 提取普通poi的特征

        pf.vid = rdd.views[i][j].vid
        pf.name = view_dict[pf.vid].name
        pf.cid = view_dict[pf.vid].cid
        pf.city = city_dict[pf.cid].name
        pf.next_city = pf.city if index[0] == len(rdd.city)-1 else city_dict[rdd.city[index[0]+1]].name
        pf.tag = "N"
        pf.start_time = get_moment_by_time(rdd.views[i][j].start_time)
        pf.end_time = get_moment_by_time(rdd.views[i][j].end_time)
        pf.dur = get_dur_desc(rdd.views[i][j].dur)
        pf.play = rdd.views[i][j].play
        pf.dining_nearby = "Y" if rdd.views[i][j].dining_nearby == 1 else "N"
        pf.traffic = (get_norm_trans_way(rdd.views[i][j].traffic[0]), get_dist_desc(rdd.views[i][j].traffic[1]), get_dur_desc(rdd.views[i][j].traffic[2]))

        pf.is_first_view = "Y" if index[1] == 0 else "N"
        pf.is_last_view = "Y" if index[1] == len(rdd.views[i]) - 1 else "N"
        pf.is_leaving_city = "N"
        
        return True

    def extract_virtual_poi_feature(pf,index,mode):
        # 给虚拟poi的特征赋值
        if mode == 1:       # 自由活动
            pf.vid = VID_VIEW
            pf.name = "自由活动"
            pf.cid = rdd.city[index]
            pf.city = city_dict[pf.cid].name
            pf.next_city = pf.city if index == len(rdd.city)-1 else city_dict[rdd.city[index+1]].name
            pf.tag = "N"
            pf.start_time = get_moment_by_time(rdd.time_by_city[i][0])
            pf.end_time = get_moment_by_time(rdd.time_by_city[i][1])
            pf.dur = get_dur_desc(rdd.time_by_city[i][2])
            pf.play = "自由活动"
            pf.is_first_view = "Y"
            pf.is_last_view = "Y"

        elif mode == 2:     # 交通离开
            pf.vid = VID_STATION    # FIXME
            pf.name = "交通站"
            pf.cid = rdd.city[index]
            pf.city = city_dict[pf.cid].name 
            pf.next_city = pf.city if index == len(rdd.city)-1 else city_dict[rdd.city[index+1]].name
            pf.tag = "S"    # FIXME
            pf.play = "乘坐交通工具离开"
            pf.is_leaving_city = "Y"

        elif mode == 3:     # 入住酒店
            pf.vid = VID_HOTEL
            pf.name = "酒店"
            pf.cid = rdd.city[index]
            pf.city = city_dict[pf.cid].name
            pf.next_city = pf.city if index == len(rdd.city)-1 else city_dict[rdd.city[index+1]].name
            pf.tag = "H"
            pf.play = "回到酒店休息"
        else:
            pass
    
        return True

    def update_poi_features():
        # 更新一些特征
        l = len(rdf.pois)
        for i in range(l):
            pf = rdf.pois[i]
            if i == 0:
                pf.is_first_poi = "Y"
            if i == l-1:
                pf.is_last_poi = "Y"
            if i > 0 and rdf.pois[i-1].cid != pf.cid:
                pf.is_first_poi = "Y"
            if i < l-1 and rdf.pois[i+1].cid != pf.cid:
                pf.is_last_poi = "Y"

        return True
            
    for i in range(len(rdd.city)):
        # 逐个加入景点，如果没有景点，但有空闲时间，加一个自由活动的景点
        if len(rdd.views[i]) == 0 and rdd.time_by_city[i][2] >= FREE_VIEW_THRESHOLD:
            pf = POIFeature()
            extract_virtual_poi_feature(pf,index=i,mode=1)
            rdf.pois.append(pf)
        else:
            for j in range(len(rdd.views[i])):
                pf = POIFeature()
                extract_poi_feature(pf,index=(i,j))
                rdf.pois.append(pf)
        # 景点加入结束后，加一个交通/酒店的虚拟点
        if rdd.city_traffic_hotel[i][0] == 1:
            pf = POIFeature()
            extract_virtual_poi_feature(pf,index=i,mode=2)
            rdf.pois.append(pf)
        elif rdd.city_traffic_hotel[i][1] == 1:
            pf = POIFeature()
            extract_virtual_poi_feature(pf,index=i,mode=3)
            rdf.pois.append(pf)
        else:
            pass
    
    update_poi_features()

    return True

def feature_parser(route, city_dict, view_dict):
    # 解析行程信息，提取行程feature, 行程数据到行程feature的映射
    # [!]随时可能重写
    assert isinstance(route, Route), "route is not an instance of Route"

    for rdd in route.data_by_day:
        rdf = RouteDayFeature()
        day_feature_parser(route, rdd, rdf, city_dict, view_dict)

        route.feature_by_day.append(rdf)

    assert len(route.data_by_day) == len(route.feature_by_day), "#feature not equal with #data"
    
    return True

def feature_reader():
    # 直接从一个json数据中，读取行程feature
    # [!]随时可能重写

    pass

if __name__ == "__main__":

    pass
