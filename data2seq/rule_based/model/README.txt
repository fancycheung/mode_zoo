模版文件简单说明

#####行程特征相关#####
feature.model  特征模版
每一行用\t分割，分别是 特征词根，特征名，可选特征值
示例：$[start_time] start_time 早上|中午|晚上

词根代表这个特征的意义，是唯一的标识符
特征名与DayRouteFeature中的某个成员变量相同，通过getattr(rdf, name) 可以取得行程特征值
可选值是这个特征词根的候选值

#####输出相关#####
corpus.model 语料模板，包括关联key，匹配模式和可选语料
用于填充输出时，具体描述关联key 的输出信息

dict.model 词典模板，包括词根和替换词
每一行用\t分割，分别是 词根，可选词
示例： $[hot_views]  热门|景点|必玩
表示hot_views在输出时，会选从可选词中选一个出来
如果entry的output中有$[hot_views]，输出词语会从替换词中选择一个（随机或某种算法）

entry.model 输出模板，包括语句间的组合和匹配条件
每一行用\n分割，不同入口间用\n\n分隔
NAME 是唯一标识符；PATTERN是入口条件；OUTPUT是输出模板
PATTERN表示，匹配所有变量&值，才可以使用这个模板，输出时，OUTPUT中的通配符会被对应值替换(有替换算法负责计算)

sentence.model 语句模板，包括句子基本结构和匹配条件
格式与entry.model相同
NAME 是唯一标识符；PATTERN是入口条件；OUTPUT是输出模板
sentence是entry子一级结构

highlight.model 行程亮点模板，包括输出语句和匹配条件
格式与entry.model相同
TODO 示例说明
是entry.model 的简化版，用于生成行程亮点。

#####各文件关系说明#####

DayRouteFeature 和 feature.model
行程特征的每个成员变量，都必须对应feature.model中的某一个feature.name
即，对一个feature.name，和一个rdf，getattr(rdf,name) 和 setattr(rdf,name) 都生效

sentence.model/highlight.model/entry.model 和 feature.model
这三个语料model中pattern部分，用$[]括起来的部分是匹配特征词根和特征词，
其中特征词根和特征词，必须在feature.model中存在

sentence.model/highlight.model/entry.model 和 dict.model
这三个语料model中output部分，用$[]括起来的部分是词典词根
必须在dict.model中存在，输出时，是选取算法从词根的可选词中选择一个输出

sentence.model和entry.model
entry是段落组织，sentence是句子组织。生成语料时，对每一个view，会生成一句sentence。这些view会选取一个合适的entry，把sentence逐句填充进去。

dict.model和feature.model
无依存关系，通配符($[])相同



