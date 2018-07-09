# textcnn  <br>
## 文件介绍 <br>
textcnn_model.py "textcnn模型" <br>
textcnn_train.py "训练teixtcnn" <br>
textcnn_predict.py "预测" <br>
tc_utils.py "textcnn utils" <br>
tc_view.py "格式化展示结果" <br>
textcnn_checkpoint_en "en checkpoint" <br>
textcnn_checkpoint_zh "zh checkpoint" <br>
tf_board "tensorboard" <br>
cache_pickle "缓存的中英文词向量、训练集、验证集" <br>

## 使用方法 <br>
文本预处理可用fasttext模型里的ft_utils.py
先使用tc_utils.py处理词向量、训练集、测试集，缓存到cache_pickle <br>
训练 "python textcnn_train.py" "修改文件内全局变量'_LANG'分别训练中英文" <br>
预测 "python textcnn_predict.py" 结果保存到result_zh.txt或result_en.txt <br>






