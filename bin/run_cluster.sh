#!/usr/bin/env bash

set -exu

input=$1
output=$2
labels=$3
dim=$4
modelname=$5

python cluster.py --input $input --output $output --labels $labels --dim $dim

java -Xmx50G -cp target/xcluster-0.1-SNAPSHOT-jar-with-dependencies.jar xcluster.eval.EvalDendrogramPurity \
--input $output \
--algorithm $modelname --dataset `basename $input` --threads 4 --print true --id-file None > $output.score