#!/bin/bash

python=python
dataset=umls
exp_dir=../exps_$dataset
exp_name=$dataset
batch_size=32
max_epoch=50
data_dir=../data/$dataset/
relation_file=$data_dir/relations.dict
gpu_id=0
cat $relation_file | awk -F "\t" '{print $2}' | while read line
do
$python -u main.py \
        --data_dir $data_dir \
        --exps_dir ./$exp_dir/ \
        --exp_name $exp_name \
        --target_relation $line \
        --batch_size $batch_size \
        --max_epoch $max_epoch \
        --step 3 \
        --length 70 \
        --use_gpu \
        --gpu_id $gpu_id
$python predict.py ./$exp_dir/$exp_name-$line-ori $batch_size 0 $gpu_id

$python predict.py ./$exp_dir/$exp_name-$line-ori $batch_size 1 $gpu_id
done
$python -u evaluate.py $data_dir $exp_dir $exp_name