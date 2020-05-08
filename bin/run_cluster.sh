#!/usr/bin/env bash

set -exu

input=$1
output=$2
labels=$3
dim=$4

python cluster.py --input $input --output $output --labels $labels --dim $dim
