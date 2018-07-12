#!/usr/bin/env python
#coding=utf-8

'''
    @desc: 根据行程特征匹配最合适的亮点模版, 主算法模块
'''

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from common import *
from common_class import *
from data_loader import load_model, load_database, load_special_model, load_virtual_POI, cross_validate
from route_parser import data_parser
from feature_parser import feature_parser
from debugger import *

# 加载静态数据
WORDS, FEATURES, CORPUS, ENTRIES, SENTENCES, HIGHLIGHTS = {},{},{},{},{},{}
CITIES, VIEWS = {},{}

# 十全大补Load
def load_all_static_data():

    if not (load_database(CITIES, VIEWS) 
            and load_virtual_POI(CITIES, VIEWS) 
            and load_model(WORDS, FEATURES, CORPUS, ENTRIES, SENTENCES, HIGHLIGHTS) 
            and load_special_model(WORDS, FEATURES, CORPUS, ENTRIES, SENTENCES, HIGHLIGHTS) 
            and cross_validate(WORDS, FEATURES, CORPUS, ENTRIES, SENTENCES, HIGHLIGHTS)):
        return False

    return True

assert load_all_static_data(), "Load Static Data Fail, Server will not start"

# 获取语料实体
def get_corpus_entity(vid):
    if CORPUS.has_key(vid):
        return CORPUS[vid]
    elif CORPUS.has_key(VID_VIEW_UTIL):
        return CORPUS[VID_VIEW_UTIL]

    return None

# 根据通配符，匹配变量值
def get_value_by_feature(feature, entites):
    # $[day->poi_sentences]      表示取poi特征，取值为poi_sentences
    # $[day->poi_sentences(0)]   表示取poi特征，取值为poi_sentences[0]
    # $[day->poi_sentences(0:-1)]   表示取poi特征，取值为从poi_sentences[0]到poi_sentences[-1]的所有值
    
    fvalue = "NULL"
    if len(feature.split('->')) != 2: return fvalue
    key, value = feature[2:][:-1].split('->')
    # TODO 考虑Word类
    if (not ENTITY_INDEX.has_key(key)) or ENTITY_INDEX[key] is None: return fvalue
    entity = entites[ENTITY_INDEX[key]]
    
    finds = INDEXPAT.findall(value)
    if len(finds) == 0:
        fvalue = str(getattr(entity, value))
    elif len(finds) == 1:
        index_str = finds[0]
        fvalue_list = getattr(entity, value.replace(index_str,''))
        indices = index_str.split(':')
        if len(indices) == 1:
            index = int(indices[0][1:-1])
            fvalue = str(fvalue_list[index])
        elif len(indices) == 2:
            fvalue = ""
            index1 = int(indices[0][1:])
            index2 = len(fvalue_list) if indices[1][:-1] == "" else int(indices[1][:-1])  
            for item in fvalue_list[index1:index2]:
                fvalue += str(item)
    
    return fvalue

# 计算实体特征与模板的匹配度
def cal_match_score(pattern, entites):
    '''
        @params: pattern: 待匹配的pattern, list of tuple of size 2
                entites: len(entites) == len(ENTITY_INDEX), 且对应位置分别为要求的实体
    '''
    # 1. 分数在0-100之间，不需要任何特征的entry给80分
    # 2. 如果有任一个属性值不能匹配，认为匹配失败，给0分
    # 3. 匹配分数 = 80 + 匹配特征数* 1, 即匹配越多的特征，分数会越高    TODO FIXME 考虑特征权重
    
    score = 80
    if len(pattern) == 0: return score
    for pat in pattern:
        value = get_value_by_feature(pat[0], entites)
        if value not in pat[1]: return 0

    score += len(pattern)

    return score

# 生成一个poi点的描述
def gen_poi_desc(route, rdf, pf):
        
    # 1. 选择poi的语料
    corp = get_corpus_entity(pf.vid)
    if corp is not None:
        corp_match_score = defaultdict(int)
        for pattern, output in corp.patterns:
            score = cal_match_score(pattern, [route,rdf,pf])
            corp_match_score[output] = score
        
        output = sorted(corp_match_score.items(), key=lambda d:d[1], reverse=True)[0][0]

        for tag in TAGPAT.findall(output):
            value = get_value_by_feature(tag, [route,rdf,pf])
            output = output.replace(tag, value)
        
        pf.corpus = output

    # 2. 选择匹配句子模板
    sentence_match_score = defaultdict(int)
    for k,v in SENTENCES.iteritems():
        score = cal_match_score(v.pattern, [route,rdf,pf])
        sentence_match_score[k] = score

    if len(sentence_match_score) == 0: return ""
    
    # 3. 生成句子
    selected_sentence = SENTENCES[sorted(sentence_match_score.items(), key=lambda d:d[1], reverse=True)[0][0]]
    sentence = selected_sentence.output
    
    for tag in TAGPAT.findall(sentence):
        value = get_value_by_feature(tag, [route,rdf,pf])
        sentence = sentence.replace(tag, value)

    return sentence

# 生成一天的行程描述
def gen_day_desc(route,rdf):
    
    # 1. 生成每个点的描述 TODO 景点聚类，生成每个团的描述 
    for pf in rdf.pois:

        sentence = gen_poi_desc(route, rdf, pf)
        rdf.poi_sentences.append(sentence)
    
    # 2. 选择整个句子的entry
    entry_match_score = defaultdict(int)
    for k,v in ENTRIES.iteritems():
        score = cal_match_score(v.pattern,[route,rdf,None])
        entry_match_score[k] = score
    
    if len(entry_match_score) == 0: return "祝您今日玩的开心"


    # 3. 逐个变量，用变量值替换, 得到最终结果
    selected_entry = ENTRIES[sorted(entry_match_score.items(), key=lambda d:d[1], reverse=True)[0][0]]
    desc = selected_entry.output

    for tag in TAGPAT.findall(desc):
        value = get_value_by_feature(tag, [route,rdf,None])
        desc = desc.replace(tag, value)
        
    # 4. 可选描述的随机选择
    for alter in ALTERPAT.findall(desc):
        alters = alter[1:][:-1].split('|')
        rp = random.choice(alters)
        desc = desc.replace(alter,rp,1)

    return desc

# 生成一天的行程亮点
def gen_day_highlight(route,rdf):
    
    return "暂未实现每日亮点功能"

# 入口
def highlight_gen(routeData):

    # step1 读取行程数据，并解析成Route
    route = Route() 
    assert data_parser(routeData, route), "Parse Route Data Failed! " 

    # step2 根据行程信息，获取每一天特征点，并选取特征点 
    assert feature_parser(route, CITIES, VIEWS), "Parse Route Feature Failed!"
    
    # step3 根据特征点，选取合适的语料(句式、词库)
    for i in range(route.days):

        rdf = route.feature_by_day[i]
        # 生成行程描述
        desc = gen_day_desc(route,rdf)
        route.desc_by_day.append(desc)

        # 生成行程亮点
        highlight = gen_day_highlight(route,rdf)
        route.highlight_by_day.append(highlight)

    # step4 组装到输出数据中
    routeData["route_desc"] = []
    for i in range(route.days):
        desc = {}
        desc["didx"] = i+1
        desc["desc"] = route.desc_by_day[i]
        desc["highlight"] = route.highlight_by_day[i]
    
    debug_info = route_debugger(route)   # 输出调试信息
    print debug_info

    return True

if __name__ == "__main__":
    
    # test
    data = json.loads(open("data/route_result.txt",'r').read().strip())["data"]
    
    highlight_gen(data)
    
    # batch test
