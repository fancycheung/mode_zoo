#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:08:56 2018

@author: miaoji
"""
import json
import os
from datetime import datetime

def str_time(str):
    return datetime.strptime(str,"%Y%m%d")
    
def handle_json(file,outfile):
    with open(file,'r') as data_f:
        json_data = json.load(data_f)
    
    spu_list = json_data['req']['data']['spu_list'][0]
    sku_list = spu_list['sku_list']
    
    for i in range(len(sku_list)):
        req = {}
        data = {}
        data["type"] = "hotel"
        
        order = {}
        order["checkin"] = spu_list["checkin"]
        order["checkout"] = spu_list["checkout"]
        order["city"] = spu_list["cid"]
        
        sku = sku_list[i]["sku"]
        data["source"] = sku["unionkey"]
        data["text"] = sku["return"]
        data["price"] = sku["price"]["val"]
        data["tax"] = sku["price"]["tax"]
        data["currency"] = sku["price"]["ccy"]
    
        data["order"] = order
        req["data"] = data
            
        json_str = json.dumps(req,ensure_ascii=False)
        
        with open(outfile,'a') as out_f:
            out_f.write(json_str+'\n')

def create_file_dir(base_dir,start_it,end_it):
    start_it = str_time(start_it)
    end_it = str_time(end_it)
    file_list = []
    for inner_dir in os.listdir(base_dir):
        if str_time(inner_dir) > start_it and str_time(inner_dir) < end_it:
            file_dir = base_dir + inner_dir + '/'
            tmp_file_list = os.listdir(file_dir)
            if tmp_file_list != None:
                tmp_file_list = [file_dir+x for x in tmp_file_list]
                file_list.extend(tmp_file_list)
    return file_list
    
    
if __name__ == "__main__":
    
    base_dir = "online/"
    outfile = "merge_json.txt"
    
    file_list = create_file_dir(base_dir,'20180121','20181024')        
    for file in file_list:
        handle_json(file,outfile)
            

