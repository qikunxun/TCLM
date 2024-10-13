#!/bin/bash

python=python3
dataset=AirGraph
exp_dir=../exps_type_$dataset
exp_name=$dataset
data_dir=../data_with_types/$dataset/
relation_file=$data_dir/relation2id.txt

beam_size=50
cat $relation_file | awk -F "\t" '{print $1}' | while read line
do
$python -u rule_extraction.py $exp_dir $exp_name $line $beam_size
done
