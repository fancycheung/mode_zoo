# **fasttext** <br>
## *文件介绍* <br>
xlsx_to_ft.py "从人工打标签的xlsx表格提取fasttext训练集、验证集。xlsx里可能会有格式错误，需要自己改" <br>
ft_utils.py "some utils" <br>
ft_model_in_py2.py "调用fasttext包主要程序，fasttext训练、预测" <br>
lang_classify.py "判断语言类别，归类相同语言" <br>
procress_data.py "中英文文本预处理" <br>
ft_utils.py "预处理、加权采样、格式化结果便于查看等方法" <br>
handle_spelling.py "处理文本拼写错误，暂时未用，速度较慢"<br>
count_vocabu.py "统计文本词汇" <br>
dict_to_label_hit.py "模板结果转为新模型fasttext要求的格式" <br>
merge_shuffle_trian_data.py "打乱样本、加权抽样、统计词汇" <br>
model_evaluate.py "评估模型测试结果，输入文件1，文件2，均为fasttext要求的格式" <br>
preprocess_for_pattern_module.py "模板模块预处理,评论按规则分句,加上评论id和分句id" <br>
pattern_module_utils.py "模板模块相关脚本" <br>

> 使用C++模板打标签流程：原始评论 --> 按规则分句,加上评论id和分句id，见preprocess_for_pattern_module.py，输出为infile --> 打包，pack_infile(infile,outfile1) --> c++模板程序(outfile1,outfile2) --> 解包，unpack_outfile(outfile2,outfile3)

> fasttext算法，整体流程：文本预处理 --> gensim库训练词向量或者导入训练好的词向量 --> 使用fasttext库训练模型 --> 导入训练好的模型，进行验证、预测

> fasttext模型，要求的文件格式,每行为,比如两个标签的评论,comment为评论:\_\_label1\_\_ \_\_label2\_\_ comment

---------

## *使用方法* 
* 同步预训练的词向量 <br>

> 从"caozhaojun@10-10-144-26:/data/caozhaojun/model_weight/fasttext/data"的"multi_ft_en_train_cbow_300d.vec"为训练好的300维英文词向量、"daodao_zh_word2vec.txt"是200维中文向量.将它们同步到本地的"./data/"目录下. 训练词向量,可以用fasttext,也可以用gensim库.

* 同步训练好的模型 <br>

> 将"caozhaojun@10-10-144-26:/data/caozhaojun/model_weight/fasttext/model"里面中英文模型同步到本地"./model/"目录下

* 文本预处理 <br>

> 文本预处理 "python ft_utils.py --inf input.txt --out output.txt --lang zh --mode 1",中文会自动用jieba进行分词.

* 加权采样 <br>

> 由于标签样本不平衡，可尝试加权采样样本用来训练模型.加权采样制作训练文本 "python ft_utils.py --inf input.txt --out output.txt --lang zh --mode 2" 

* 训练、预测fasttext <br>

> 训练 "python2.7 ft_model_in_py2.py --train 1 --lang zh",fasttext安装在python2.7上,若是python3需要修改代码.训练集、验证集在代码里写死了,可以在里面修改

> 预测 "python2.7 ft_model_in_py2.py --train 0 --lang zh --gate 0.1" 

> 格式化结果，便于查看 "python ft_utils.py --inf input.txt --mode 3"


## *酒店评论数据集* <br>
* 描述 <br>

> "caozhaojun@10-10-144-26:/data/caozhaojun/model_weight/merge_source_id"目录下,文档是从原始酒店评论目录下汇总而来，
一个源一个文件."comment_len_pickle"为各源各语言字词长度统计."split_lang\_\*"为最终四个源，按语言归类.目前使用的是daodao数据集，其他未用。见caozhaojun@10-10-144-26:/data/caozhaojun/model_weight/daodao_dataset。目前让标注人员标注的是procressed_daodao_en_shuffle.txt、procressed_daodao_zh_shuffle.txt,每个文件的最后1万条评论。






