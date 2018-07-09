# fasttext <br>
## 文件介绍<br>
xlsx_to_ft.py "xlsx表格提取fasttext训练集、验证集" <br>
ft_utils.py "some utils" <br>
ft_model_in_py2.py "fasttext训练、预测" <br>
lang_classify.py "判断语言，归类相同语言" <br>
procress_data.py "中英文文本预处理" <br>
ft_utils.py "预处理、加权采样、格式化结果便于查看等方法" <br>
handle_spelling.py "处理文本拼写错误"<br>
count_vocabu.py "统计文本词汇" <br>
dict_to_label_hit.py "模板结果转为新模型要求的格式" <br>
merge_shuffle_trian_data.py "打乱样本、加权抽样、统计词汇" <br>
model_evaluate.py "评估模型测试结果" <br>
preprocess_for_pattern_module.py "模板模块预处理" <br>
pattern_module_utils.py "模板模块相关脚本" <br>

## 使用方法
文本预处理 "python ft_utils.py --inf input.txt --out output.txt --lang zh --mode 1" <br>
加权采样制作训练文本 "python ft_utils.py --inf input.txt --out output.txt --lang zh --mode 2" <br>
训练 "python2.7 ft_model_in_py2.py --train 1 --lang zh" <br>
预测 "python2.7 ft_model_in_py2.py --train 0 --lang zh --gate 0.1" <br>
格式化结果，便于查看 "python ft_utils.py --inf input.txt --mode 3" <br>









