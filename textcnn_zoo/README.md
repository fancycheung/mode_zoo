# **textcnn**  <br>
## *文件介绍* <br>
textcnn_model.py "textcnn模型" <br>
textcnn_train.py "训练textcnn" <br>
textcnn_predict.py "预测" <br>
tc_utils.py "textcnn utils" <br>
tc_view.py "格式化展示结果" <br>
textcnn_checkpoint_en "en checkpoint" <br>
textcnn_checkpoint_zh "zh checkpoint" <br>
tf_board "tensorboard" <br>
cache_pickle "缓存的中英文词向量、训练集、验证集" <br>
textrcnn_model.py "textrcnn模型,完成多标签测试，需要按照textcnn相关内容，补充textrcnn_train.py,textrcnn_predict.py"
textrnn_model.py "textrnn模型,完成单一标签测试，需要按照textcnn相关内容，完成多标签测试，并补充textrcnn_train.py,textrcnn_predict.py"

----

## *使用方法* 
* 同步词向量、训练好的模型 <br>

> 同步"caozhaojun@10-10-144-26:/data/caozhaojun/model_weight/textcnn/cache_pickle"下的词向量、训练集、验证集等缓存到本地"./cache_pickle/"目录下 

> 同步"caozhaojun@10-10-144-26:/data/caozhaojun/model_weight/textcnn/textcnn_checkpoint_zh"到本地"./textcnn_checkpoint_zh/" 

> 同步"caozhaojun@10-10-144-26:/data/caozhaojun/model_weight/textcnn/textcnn_checkpoint_en"到本地"./textcnn_checkpoint_en/"

* 文本预处理 <br>

> 文本预处理可用fasttext模型里的ft\_utils.py

* 缓存词向量、训练集、验证集 <br>

> 对新的数据集，可以用训练好的词向量,使用tc\_utils.py处理词向量、训练集、验证集缓存到cache\_pickle,方便模型使用

* 训练、预测 <br>

> 训练 "python textcnn\_train.py" "修改文件内全局变量'\_LANG'分别训练中英文" 

> 预测 "python textcnn\_predict.py" 结果保存到result_zh.txt或result_en.txt 






