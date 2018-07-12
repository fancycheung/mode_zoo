#!/usr/bin/env python
#coding=utf-8

'''
    @desc: 进入mongodb读取行程数据
'''

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import re
import os
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pymongo import MongoClient
from copy import deepcopy

TRAFFIC_MODE = defaultdict(str)
TRAFFIC_MODE[10001] = "flight"
TRAFFIC_MODE[10002] = "train"
TRAFFIC_MODE[10003] = "bus"
TRAFFIC_MODE[10004] = "drive"
TRAFFIC_MODE[10005] = "drive"

# 连接route数据库
def RouteMongoConn():

    db = MongoClient("mongodb://writer:miaoji1109@10.10.233.135")["store"]
    db.authenticate("writer","miaoji1109")

    collection = db.get_collection("session")  

    return collection

# 获取所有行程数据
def get_all_route():
    
    collection = RouteMongoConn()
    with open('./data/all_route','w') as w:
        for find in collection.find({"route":{"$exists":True}}):
            w.write(json.dumps(find,ensure_ascii=False) + '\n')

    return True

# 为所有行程生成route_day
def gen_all_route_day():
    
    index = 0
    with open('./data/all_route_day','w') as w:
        for line in open('./data/all_route','r'):
            if line.strip() == "":continue

            index += 1

            data= json.loads(line.strip())
            try:
                gen_route_day(data)
            except Exception,e:
                print '%s route error: %s'%(index, str(e))
                continue

            w.write(json.dumps(data,ensure_ascii=False) + '\n')
    
    return True

# 为一个行程生成route_day
def gen_route_day(data):

    routeJ = data["route"]
    summaryJ = data["summary"]
    productJ = data["product"]
    routeDayJ = []

    dept_date = summaryJ["dept_date"]
    days = summaryJ["day"]

    def init_route_day():
        # 根据日期初始化所有route_day
        
        for d in range(0,days):
            date = (datetime.strptime(dept_date,"%Y%m%d")+timedelta(days=d)).strftime("%Y%m%d_%H").split('_')[0]
            single_day = {}
            single_day["date"] = date
            single_day["special"] = 0   # 无用字段
            single_day["one_the_way"] = 0   # 会被重置
            single_day["week"] = -1 # 无用字段
            single_day["didx"] = d+1
            single_day["highlight"] = {}    # 无用字段

            single_day["city_show"] = []
            single_day["city"] = []

            routeDayJ.append(single_day)
        
        return True

    def get_city_day(index, city_route):
        # 把城市的所有天，逐个放到对应的route_day里
        city_start_date = city_route["arv_time"].split("_")[0]
        city_end_date = city_route["dept_time"].split("_")[0]
        city_util = {"cid":city_route["cid"],"ridx":index,"traffic":""}
        city_show_util = {"cid":city_route["cid"],"ridx":index,"checkin":city_route["checkin"],"checkout":city_route["checkout"],\
                "need_traffic":0, "need_hotel":0,"time":{"from":"00:00","to":"23:59"},"product":{}}

        didx, today = 0, city_start_date
        while(today <= city_end_date):
            route_day = None
            for rd in routeDayJ:
                if rd["date"] == today:
                    route_day = rd
                    break
            
            #assert route_day is not None, "fuck bug"
            if route_day is None: 
                today = (datetime.strptime(today,"%Y%m%d")+timedelta(days=1)).strftime("%Y%m%d_%H").split('_')[0]
                didx += 1
                continue

            city = deepcopy(city_util)
            city_show = deepcopy(city_show_util)
            
            # 计算必填特征
            if today == city_start_date:
                city_show["time"]["from"] = city_route["arv_time"].split("_")[1]

            if today == city_end_date:
                city_show["need_traffic"] = 1
                city_show["time"]["to"] = city_route["dept_time"].split("_")[1]

                if city_route["traffic"] is not None and city_route["traffic"].has_key("product_id"):
                    product_id = city_route["traffic"]["product_id"]
                    city["traffic"] = TRAFFIC_MODE[productJ["traffic"][product_id]["mode"]]

            if today >= city_start_date and today < city_end_date:
                city_show["need_hotel"] = 1

            if (index ==0 or index == len(routeJ)-1) and (today < city_end_date and today > city_start_date):
                city_show["on_the_way" ] = 1
            
            city_show["view"] = None
            if city_route["view"] is not None and len(city_route["view"]["day"]) > didx:
                city_show["view"] = {}
                city_show["view"]["didx"] = didx
                city_show["view"]["day"] = deepcopy(city_route["view"]["day"][didx])
                city_show["view"]["day"]["baoche"] = 0  # 应该有逻辑赋值，但未找到依存字段
                view_list = deepcopy(city_show["view"]["day"]["view"])
                city_show["view"]["day"]["view"] = []
                for item in view_list:
                    if item["type"] != 2: continue
                    city_show["view"]["day"]["view"].append(item)

            route_day["city"].append(city_show)
            route_day["city_show"].append(city)     # 前面的city、city_show命名反了，但我懒得改了

            today = (datetime.strptime(today,"%Y%m%d")+timedelta(days=1)).strftime("%Y%m%d_%H").split('_')[0]
            didx += 1
            

        return True

    init_route_day()

    for i in range(len(routeJ)):
        get_city_day(i,routeJ[i])

    data["route_day"] = routeDayJ

    return True

if __name__ == "__main__":
    #get_all_route()

    gen_all_route_day()
