#!/bin/bash

Q_GPU=memory.total,memory.used
FORMAT=csv,noheader,nounits

nvidia-smi --format=${FORMAT} --query-gpu=${Q_GPU} | \
    while read -r line
    do
        info=`echo $line | awk -F", " '{print $2/$1*100}'`
        echo "f={$info}"
    done
