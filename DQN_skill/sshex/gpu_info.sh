#!/bin/bash

Q_GPU=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free
FORMAT=csv,noheader,nounits
FIELD_TYPE=("i" "s" "f" "f" "i" "i" "i")

echo "headers={$Q_GPU}"
nvidia-smi --format=${FORMAT} --query-gpu=${Q_GPU} | \
    while read -r line ; do
        IFS=$',' read -r -a info <<< "$line"
        for ((i=0;i<${#info[@]};i++)); do 
            printf "${FIELD_TYPE[$i]}={$(echo ${info[$i]} | sed 's/^[ \t]*//')}";
        done
        printf "\n"
    done

