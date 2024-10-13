#!/bin/bash

python=python3
dataset=AirGraph
exp_dir=../exps_type_$dataset
exp_name=$dataset
batch_size=4
max_epoch=50
data_dir=../data_with_types/$dataset/
relation_file=$data_dir/relation2id.txt
gpu_id=0
cat $relation_file | awk -F "\t" '{print $1}' | while read line
do
$python -u main_type.py \
        --data_dir $data_dir \
        --exps_dir ./$exp_dir/ \
        --exp_name $exp_name \
        --target_relation $line \
        --batch_size $batch_size \
        --max_epoch $max_epoch \
        --use_gpu \
        --gpu_id $gpu_id \
        --step 3
$python predict_type.py ./$exp_dir/$exp_name-${line//\//\|}-ori $batch_size 0 $gpu_id

$python predict_type.py ./$exp_dir/$exp_name-${line//\//\|}-ori $batch_size 1 $gpu_id
done
$python -u evaluate_type.py $data_dir $exp_dir $exp_name
