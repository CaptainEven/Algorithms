#!/bin/bash

function addArray
{
    local sum=0
    local new_array

    # 解析传入函数的命令行参数
    new_array=($(echo "$@"))

    for value in ${new_array[*]}
    do
        sum=$[ $sum + $value ]
    done

    # 函数返回单个数值
    echo $sum
}

function doubleArray
{
    local ori_array
    local new_array
    local array_size
    local i

    # 解析传入函数的所有命令行参数
    ori_array=($(echo "$@"))
    new_array=($(echo "$@"))

    # 解析传入函数命令行的参数个数$#
    # array_size=$[ $# ]
    array_size=$#  

    # echo "The input array has $array_size elements"
    # echo

    for (( i=0; i<array_size; ++i ))
    {
        new_array[$i]=$[ ${ori_array[i]} * 2 ]
    }

    # 函数返回数组的所有元素
    echo ${new_array[*]}
}

# 使用函数递归: 函数调用自身
# 传入函数的第一个命令行参数$1
function factorial
{
    if [ $1 -eq 1 ]
    then
        echo 1
    else
        local tmp=$[ $1 - 1 ] 
        local result=$(factorial $tmp)
        echo $[ $result * $1 ] 
    fi 
}

# -------------------测试函数调用

# # 原始数组
# input_array=(1 2 3 4 5 6)
# echo "The original array is: ${input_array[*]}"

# # 构建参数
# arg1=$(echo ${input_array[*]})
# echo "arg1: $arg1"

# # 将参数传入函数命令行参数
# # result=$(addArray $arg1)

# # 单独向函数传入数组名变量是不行的
# result=$(addArray $input_array)

# # 需要传入数组的一系列元素${input_array[*]}
# # 函数返回一个值
# result=$(addArray ${input_array[*]}) 
# echo "The result is $result"

# # 函数返回数组的所有元素
# result=($(doubleArray ${input_array[*]}))
# echo "The result is ${result[*]}"

# read -p "Enter integer value: " value
# result=$(factorial $value)
# echo "The factorial of $value is: $result"

# -------------------
