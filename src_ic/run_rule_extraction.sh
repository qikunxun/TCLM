#!/bin/bash

python=python3
dataset=umls
exp_dir=../exps_$dataset
exp_name=$dataset
data_dir=../data/$dataset/
relation_file=$data_dir/relations.dict

beam_size=20
cat $relation_file | awk -F "\t" '{print $2}' | while read line
do
$python -u rule_extraction.py $exp_dir $exp_name $line $beam_size
done
