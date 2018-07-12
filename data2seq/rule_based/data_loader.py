#!/usr/bin/env python
#coding=utf-8

'''
    @desc: 加载模型文件，包括：数据库，本地模版模型
'''

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from common import *
from common_class import *
from route_parser import city_parser, view_parser
import db

def load_once():
    # 读数据库速度受限，只读一次，将数据写入本地文件

    fpath = "./model/intro.model"

    with open(fpath,'w') as w:
        for item in db.QueryBySQL(LABELDATA_IP, LABELDATA_DB, "select * from %s where 1"%CONTENT_TABLE):
            vid = item["poi_id"]
            content = json.loads(item["content"])
            name = content["cn"]["name_cn"]["val"]
            intro = content["cn"]["modifier_prefix"]["val"] + name + content["cn"]["modifier_suffix"]["val"]

            w.write(vid + '\t' + name + '\t' + intro + '\n')

    return True

def load_database(cities, views):
    # 加载数据库数据
    tags = defaultdict(str)
    countries = defaultdict(str)
    city_view_num = defaultdict(int)

    # 读取tag表
    print "加载tag信息..."
    for item in db.QueryBySQL(BASEDATA_IP, BASEDATA_DB, "select * from tag where 1"):
        tags[item["tag_id"]] = item["tag"]
    print "加载tag信息成功，共加载%d个tag"%len(tags)
    
    # 读取country表
    print "加载country信息..."
    for item in db.QueryBySQL(BASEDATA_IP, BASEDATA_DB, "select * from country where 1"):
        countries[item["mid"]] = item["name"]
    print "加载country信息成功，共加载%d个country"%len(countries)

    # 读取attraction表
    attr_infos = db.QueryBySQL(BASEDATA_IP, BASEDATA_DB, "select id, name, name_en, alias, map_info, city_id, ranking, intensity, tag, status_online, status_test from attraction where 1")
    print "读取attraction表成功，共读取%d个attraction"%len(attr_infos)

    for item in attr_infos:
        city_view_num[item["city_id"]] += 1

    # 读取city表, 并加载city信息
    print "加载city信息..."
    for item in db.QueryBySQL(BASEDATA_IP, BASEDATA_DB, "select * from city where status_online='Open' or status_test='Open' or dept_status_online='Open' or dept_status_test='Open' "):
        city = City()
        city_parser(item, city)
        city.view_num = city_view_num[city.cid]
        city.country = countries[city.mid]

        cities[city.cid] = city

    print "加载city信息成功，共加载%d个city"%len(cities)

    # 加载attraction信息
    print "加载view信息..."
    # 先读取本地intro.model
    intros = defaultdict(str)
    for line in open('./model/intro.model','r'):
        items = line.strip().split('\t')
        if len(items) != 3: continue
        intros[items[0].strip()] = items[2].strip()

    for item in attr_infos:
        if item["status_online"] != 'Open' and item["status_test"] != 'Open': continue
        if len(item["map_info"].split(',')) != 2: continue
        if not cities.has_key(item["city_id"]): continue

        view = View()
        view_parser(item, view)
        
        if item["tag"] != "NULL":
            for tag_id in item["tag"].split('|'):
                view.tags.append(tags[tag_id])
    
        view.rank_region = int(float(view.rank) / cities[view.cid].view_num * 100)

        # 补充corpus
        if intros[view.vid] == "" or intros[view.vid] == "NULL" or intros[view.vid] == view.name:
            city = None
            if cities.has_key(view.cid):
                city = cities[view.cid]
            view.intro = get_util_intro(city,view)
        else:
            view.intro = intros[view.vid]
        
        views[view.vid] = view
    
    print "加载view信息成功，共加载%d个view"%len(views)

    return True

def load_virtual_POI(cities, views):
    
    # 加载虚拟attraction
    print "加载虚拟POI"
    vCityCenter = {"type_tag":"C", "vid": VID_CITY, "name": "市中心区域"}
    vHotel = {"type_tag":"H", "vid": VID_HOTEL, "name": "下榻酒店"}
    vAirport = {"type_tag":"A", "vid": VID_AIRPORT, "name": "机场"}
    vStation = {"type_tag":"S", "vid": VID_STATION, "name": "车站"}
    vStation = {"type_tag":"V", "vid": VID_VIEW, "name": "自由活动"}

    for data in [vCityCenter, vHotel, vAirport, vStation]:
        poi = View()
        for k, v in data.iteritems():
            setattr(poi,k,v)

        views[poi.vid] = poi
    
    print "加载虚拟POI 完成"
    return True
    
def load_model(words, features, corpus, entries, sentences, highlights):
    # 加载模型文件数据
    print "加载dict.model..."
    for line in open('model/dict.model','r'):
        if line.strip() == "" or line.startswith('#'):
            continue
        
        word = Word()
        if not word.parse(line.strip()):
            print "加载word出错,原文[%s]"%line.strip()
            continue
        words[word.name] = word

    print "加载dict.model完成，共加载%s个word"%len(words)

    print "加载feature.model..."
    for line in open('model/feature.model','r'):
        if line.strip() == "" or line.startswith('#'):
            continue

        feature = Feature()
        if not feature.parse(line.strip()):
            print "加载feature出错,原文[%s]"%line.strip()
            continue
        features[feature.name] = feature

    print "加载feature.model完成，共加载%s个feature"%len(features)
    
    print "加载corpus.model..."
    for line in open('model/corpus.model','r'):
        if line.strip() == "" or line.startswith('#'):
            continue
        
        corp_id = line.strip().split('\t')[0]
        if corpus.has_key(corp_id):
            corp = corpus[corp_id]
            if not corp.add_pattern(line):
                print "加载corpus出错,原文[%s]"%line.strip()
                continue
        else:
            corp = Corpus()
            corp.vid = corp_id
            corpus[corp.vid] = corp
            if not corp.add_pattern(line):
                print "加载corpus出错,原文[%s]"%line.strip()
                continue

    print "加载corpus.model完成，共加载%s个corpus"%len(corpus)

    print "加载entry.model..."
    for item in open('model/entry.model','r').read().strip().split('\n\n'):
        rows = item.strip().split('\n')
        
        entry = Entry()
        if not entry.parse(rows):
            print "加载entry出错,原文[%s]"%item.strip()
            continue
        entries[entry.name] = entry

    print "加载entry完成，共加载%s个entry"%len(entries)
    
    print "加载sentence.model..."
    for item in open('model/sentence.model','r').read().strip().split('\n\n'):
        rows = item.strip().split('\n')

        sentence = Entry()
        if not sentence.parse(rows):
            print "加载sentence出错,原文[%s]"%item.strip()
            continue
        sentences[sentence.name] = sentence

    print "加载sentence完成，共加载%s个sentence"%len(sentences)
    
    print "加载highlight.model..."
    for item in open('model/highlight.model','r').read().strip().split('\n\n'):
        rows = item.strip().split('\n')

        highlight = Entry()
        if not highlight.parse(rows):
            print "加载highlight出错,原文[%s]"%item.strip()
            continue
        highlights[highlight.name] = highlight

    print "加载highlight完成，共加载%s个highlight"%len(highlights)

    return True

def load_special_model(words, features, corpus, entries, sentences, highlights):

    # 加载特殊词
    for w in SPECIAL_WORDS:
        word = Word()
        word.name = w
        word.root = w[2:][:-1]
        word.tag = "S"
        words[word.name] = word
    
    # 加载特殊语料
    pass

    return True

def debug_key_values(words, features, corpus, entries, sentences, highlights):
    # 输出所有出现的特征，特征值，便于补充各实体的feature
    kvs = defaultdict(list)
    # entries, sentences, highlights
    for targets in [entries, sentences, highlights]:
        for k,v in targets.iteritems():
            for pat in v.pattern:
                key = pat[0]
                for find in INDEXPAT.findall(key): key = key.replace(find,'')
                kvs[key].extend(pat[1])

            for key in TAGPAT.findall(v.output):
                for find in INDEXPAT.findall(key): key = key.replace(find,'')
                kvs[key].extend([])

    for k,v in kvs.iteritems():
        print k + '\t' + '|'.join(list(set(v)))

    return True

def cross_validate(words, features, corpus, entries, sentences, highlights):
    

    # 验证各属性间关系，保证后面的程序不会取错变量
    print "检测所有entry中，feature都能在对应实体取到"

    rdf = RouteDayFeature()
    pf = POIFeature()
    route = Route()

    entites = [route,rdf,pf]
    
    def check_attr(name, tag):
        if len(tag.split('->')) != 2:
            print "实体%s，变量命名%s不符合规则"%(name,tag)
            return False

        key, value = tag[2:][:-1].split('->')
        for find in INDEXPAT.findall(value):
            value = value.replace(find,"")
        entity = entites[ENTITY_INDEX[key]]
        try:
            getattr(entity, value)
        except:
            print "实体%s，无法取到%s属性"%(name, value)
            return False

        return True

    for entries in [entries, sentences, highlights]:
        for name, entity in entries.iteritems():
            for tag in TAGPAT.findall(entity.output):
                if not check_attr(name,tag): return False
            for tv in entity.pattern:
                if not check_attr(name,tv[0]): return False

    for name, entity in corpus.iteritems():
        for pattern in entity.patterns:
            for tag in TAGPAT.findall(pattern[1]):
                if not check_attr(name, tag): return False
            for tv in pattern[0]:
                if not check_attr(name,tv[0]): return False

    print "检测成功"
    return True # FIXME

    # TODO 继续检测其他属性和依存关系

    # 检验feature和word存在
    for k, v in entries.iteritems():
        # 检查output的所有N词，都在dict中
        for w in TAGPAT.findall(v.output):

            if w not in SPECIAL_WORDS and not words.has_key(w):
                print "Entry %s 检测output中词%s不在dict.model中"%(k,w)
                return False
        # 检查pattern 的合法性
        for p in v.pattern:
            if not features.has_key(p[0]):
                print "Entry %s 检测pattern中特征%s不在feature.model中"%(k,p[0])
                return False
            for w in p[1]:
                if w not in features[p[0]].values:
                    print "Entry %s 检测pattern中特征值%s不在feature %s的value中, 合法值必须在%s中"%(k,w,p[0],features[p[0]].values)
                    return False
    
    for k,v in features.iteritems():
        try:
            _ = getattr(rdf, v.key)
        except:
            print "在RouteDayFeature中无法取到feature %s"%k
            return False

    for k,v in words.iteritems():
        if v.tag == "N": continue
        try:
            _ = getattr(rdf, v.root)
        except:
            print "在RouteDayFeature中无法取到special word %s"%k
            return False

    print "验证成功"

    return True

if __name__ == "__main__":
    
    # test
    words, features, entries, sentences, highlights, corpus = {},{},{},{},{},{}
    load_model(words, features, corpus, entries, sentences, highlights)
    load_special_model(words, features, corpus, entries, sentences, highlights)
    #debug_key_values(words, features, corpus, entries, sentences, highlights)   # 打印所有特征/值
    cross_validate(words, features, corpus, entries, sentences, highlights)

    cities, views = {}, {}
    load_database(cities, views)       # 耗时较长 不要随便测试
    load_virtual_POI(cities, views)

    #load_once()    # 只需运行一次

