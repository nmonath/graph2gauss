#!/usr/bin/env bash

set -exu

input=$1
output=$2
labels=$3
dim=$4
modelname=$5

python cluster.py --input $input --output $output --labels $labels --dim $dim
java -Xmx50G -cp target/xcluster-0.1-SNAPSHOT-jar-with-dependencies.jar xcluster.eval.EvalDendrogramPurity \
--input ../graph2gauss/exp_out/g2g-100/aloi.5k.new.graph.5.npz/2020-05-08-10-26-03/tree.tsv \
--algorithm $modelname --dataset `basename $input` --threads 4 --print true --id-file None > $output.score