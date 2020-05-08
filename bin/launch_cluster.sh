#!/usr/bin/env bash

set -exu

input=$1
output=$2
labels=$3
dim=$4

partition=${5:-gpu}
mem=${6:-12000}
threads=${7:-4}
gpus=${8:-1}

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

dataset=`basename $input`
model_name="g2g-$dim"
job_name="$model_name-$dataset-$TIME"
log_dir=logs/$model_name/$dataset/$TIME
log_base=$log_dir/$job_name
out_dir=exp_out/$model_name/$dataset/$TIME
mkdir -p $log_dir
mkdir -p $out_dir


sbatch -J $job_name \
            -e $log_base.err \
            -o $log_base.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --gres=gpu:$gpus \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --time=0-04:00 \
            bin/run_cluster.sh $input $out_dir/tree.tsv $labels $dim