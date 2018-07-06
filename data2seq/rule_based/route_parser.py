#!/usr/bin/env python
#coding=utf-8

'''
    @desc: 解析行程数据，并录入到Route类中
'''

from common import *
from common_class import *

def city_parser(data, city):
    # 解析city数据
    assert isinstance(data, dict), "data is not a dict"
    assert isinstance(city, City), "city is not an instance of City"

    city.cid = data["id"]
    city.mid = data["country_id"]
    city.name = data["name"]
    city.name_en = data["name_en"]
    #city.alias = []
    city.mapinfo = (float(data["map_info"].split(',')[0].strip()), float(data["map_info"].split(',')[1].strip()))

    city.is_park = data["is_park"]
    city.grade = data["grade"]

    return 0

def view_parser(data, view):
    # 解析view数据
    assert isinstance(data, dict), "data is not a dict"
    assert isinstance(view, View), "view is not an instance of View"

    view.vid = data["id"]
    view.name = data["name"]
    view.name_en = data["name_en"]
    if(data["alias"].strip() != ""): view.alias = data["alias"].strip().split('|')
    view.cid = data["city_id"]
    view.mapinfo = (float(data["map_info"].split(',')[0].strip()), float(data["map_info"].split(',')[1].strip()))
    view.type_tag = "N"
    view.rank = data["ranking"]
    # tags, rank_region 值无法给出
    if data["intensity"] != "NULL":
        rcmd_level, rcmd_time = -1, 0
        rcmd_way = "NULL"
        for item in json.loads(data["intensity"]):
            if item["recommend"] > rcmd_level:
                rcmd_level = item["recommend"]
                rcmd_way = item["play"]
                rcmd_time = item["time_length"]["day"] * 1440 + item["time_length"]["hour"] * 60 + item["time_length"]["minute"]

        view.play_rcmd = (rcmd_way, rcmd_time)
        
    return 0

def day_data_parser(rdd, day_data, route):
    # 解析按天的数据

    rdd.date = day_data["date"]
    rdd.on_the_way = day_data["on_the_way"]
    rdd.day_index = days_delta(route.dept_date, rdd.date)

    def extract_single_poi_data(poi,data):
        # 提取一个poi的数据
        poi.vid = data["id"]
        poi.start_time = cal_fen(data["stime"].split('_')[1])
        poi.end_time = cal_fen(data["etime"].split('_')[1])
        poi.dur = poi.end_time - poi.start_time
        poi.do_what = data["do_what"]
        poi.play = data["play"]
        poi.dining_nearby = data["dining_nearby"]
        if data["traffic"] is not None:
            poi.traffic = (data["traffic"]["type"], data["traffic"]["dist"], data["traffic"]["dur"]/60)
        
        return True

    def extract_single_city_data(day_data, city_data):

        # 提取一个城市的数据
        city = city_data["cid"]
        city_index = city_data["ridx"]
        city_traffic_hotel = (city_data["need_traffic"], city_data["need_hotel"])
        if city_data["view"] is not None:
            views_data = city_data["view"]["day"]
            city_dur = (cal_fen(views_data["stime"].split('_')[1]), cal_fen(views_data["etime"].split('_')[1]), \
                    minutes_delta(views_data["stime"], views_data["etime"]))
            city_day_index = city_data["view"]["didx"]
        
            view_data = views_data["view"]
            views = []
            # 逐个添加当天安排点 XXX 不考虑前一天最后的点
            if len(view_data) > 0:
                for s_view_data in view_data:
                    poi = POIData()
                    extract_single_poi_data(poi, s_view_data)
                    views.append(poi)
        else:   # 出发或返回城市会没有view结构
            city_dur = (60,120,60)      # XXX 随意赋值
            city_day_index = 0
            views = []

        return city, city_dur, city_day_index, city_index, city_traffic_hotel, views

    for city_data in day_data["city"]:
        city, city_dur, city_day_index, city_index, city_traffic_hotel, views = extract_single_city_data(day_data,city_data)

        rdd.city.append(city)
        rdd.time_by_city.append(city_dur)
        rdd.city_day_index.append(city_day_index)
        rdd.city_index.append(city_index)
        rdd.city_traffic_hotel.append(city_traffic_hotel)
        rdd.views.append(views)

    return True

def data_parser(data, route):
    # 解析行程Json，提取行程信息
    # [!]随时可能重写
    assert isinstance(route, Route), "route is not an instance of Route"
    
    routeJ = data["route"]
    productJ = data["product"]
    summaryJ = data["summary"]
    routeDayJ = data["route_day"]
    
    # 基本信息解析
    route.dept_date = summaryJ["dept_date"]
    route.back_date = summaryJ["back_date"]
    route.days = len(routeDayJ)
    dept_time = summaryJ["dept_time"].split("_")[1]
    route.dept_time = cal_fen(dept_time)
    back_time = routeJ[-1]["arv_time"].split("_")[1]
    route.back_time = cal_fen(back_time)

    route.city_num = len(routeJ)
    route.dept_city = routeJ[0]["cid"]
    route.dept_city = routeJ[-1]["cid"]

    route.title = summaryJ["title"]

    # 按天解析
    for rdj in routeDayJ:
        rdd = RouteDayData()
        day_data_parser(rdd, rdj, route)

        route.data_by_day.append(rdd)

    return True

if __name__ == "__main__":

    pass
