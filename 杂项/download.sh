#!/bin/bash

# 从huggingface下载模型和数据集的脚步
# 使用方法：
# 下载模型：bash download.sh -m model_name
# 下载数据集：bash download.sh -d dataset_name
# 下载模型和数据集：bash download.sh -m model_name -d dataset_name


# Initialize our own variables
model_name=""
dataset_name=""
export HF_ENDPOINT=https://hf-mirror.com
# Parsing options
while getopts ":m:d:" opt; do
  case $opt in
    m)
      model_name="$OPTARG"
      ;;
    d)
      dataset_name="$OPTARG"
      ;;
    \?)
      echo "Invalid option -$OPTARG" >&2
      ;;
  esac
done

# Shift off the options and optional --.
shift $((OPTIND-1))

# Check positional arguments
if [ -n "$1" ]; then
  model_name="$1"
fi

if [ -n "$2" ]; then
  dataset_name="$2"
fi

# Display the variables
# echo "Model name: $model_name"
# echo "Dataset name: $dataset_name"
# =$1

if [ ${#model_name} -gt 0 ]
then
  echo "Model name: $model_name"
  huggingface-cli download --resume-download ${model_name} --local-dir model/${model_name}
else
  echo "Model name is not set or empty"
fi

if [ ${#dataset_name} -gt 0 ]
then
  echo "Dataset name: $dataset_name"
  huggingface-cli download --repo-type dataset --resume-download ${dataset_name} --local-dir dataset/${dataset_name}
else
  echo "Dataset name is not set or empty"
fi
# if $model_name
#