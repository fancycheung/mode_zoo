#!/bin/bash
for lg in 'zh' 'en'
do
	for var in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
	do
		python2.7 ft_model_in_py2.py --lang $lg --gate $var
		echo $lg $var
	done
done

python model_evaluate.py

