set -ex


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export VLLM_LOGGING_LEVEL=DEBUG

# PROMPT_TYPE=qwen-boxed
MODEL_NAME_OR_PATH=$1
# OUTPUT_DIR=$1
SEED=$2
n_sampling=$3

SPLIT="test"
NUM_TEST_SAMPLE=-1

MAX_GEN_TOKENS=${4:-4096}
DATA_NAME=${5:-"math_500,math,gsm8k,train_amc_aime,aime24,amc23"}
PROMPT_TYPE=${6:-"qwen-boxed"}
OUTPUT_DIR=${7:-$MODEL_NAME_OR_PATH}

# English open datasets
# DATA_NAME="math_500,math,gsm8k,train_amc_aime,aime24,amc23"

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature 0.6 \
    --n_sampling $n_sampling \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --max_tokens_per_call=$MAX_GEN_TOKENS \
    --data_parallel_size 2 \
    --save_outputs \
    # --overwrite \

ray stop
exit
