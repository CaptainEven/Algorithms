#!/bin/bash

# ------------ 对指定后缀的文件批量重
root=$1
pre_suffix=$2
cur_suffix=$3

if [ $@ -ne 3 ]
then
    echo "[Err]: wrong number of parameters."
fi

for file in $root/*
do
    if [ ! -f $file ]
    then
        continue
    fi

    # echo "processing $file..."
    new_file=$(echo $file | sed 's/\.sh/\.bash/')
    # echo "new_file: $new_file"

    mv $file $new_file
    rm -rf $file
done

