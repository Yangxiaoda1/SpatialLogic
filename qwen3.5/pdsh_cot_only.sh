#!/bin/bash

# 多节点分布式训练启动脚本
export PDSH_RCMD_TYPE=ssh

# 解析逗号分隔的 NODE_IP_LIST，例如 ip1:8,ip2:8。
if [ -z "${NODE_IP_LIST:-}" ]; then
    echo "NODE_IP_LIST is empty" >&2
    exit 1
fi
IFS=',' read -r -a node_specs <<< "$NODE_IP_LIST"
node_ips=()
for spec in "${node_specs[@]}"; do
    node_ips+=("${spec%%:*}")
done

# 设置主节点
export MASTER_ADDR=${node_ips[0]}

# 集群范围选择统一可用端口（随机起点 + 跨节点实际 bind 测试）
get_cluster_free_port() {
    local start_port=${1}
    if [ -z "$start_port" ]; then
        start_port=$((36000 + RANDOM % 20000))
    fi
    local port=$start_port
    while true; do
        # 仅在主节点检测端口是否被占用（连接式检测：若能连接说明已被占用）
        # 分布式只在 MASTER 上监听该端口，因此只需确保 MASTER 节点空闲
        local check_output
        check_output=$(pdsh -w "$MASTER_ADDR" "python3 -c \"import socket as s; p=$port; so=s.socket(s.AF_INET, s.SOCK_STREAM); so.settimeout(0.5); r=so.connect_ex(('127.0.0.1', p)); so.close(); print('used' if r==0 else 'free')\"" 2>/dev/null)
        if ! echo "$check_output" | grep -q used; then
            echo $port
            return 0
        fi
        port=$((port+1))
        if [ $port -gt 65000 ]; then
            port=36000
        fi
    done
}

export MASTER_PORT=$(get_cluster_free_port 36000)
echo "[pdsh.sh] Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

SCRIPT_DIR="/apdcephfs_gy2_302507476/share_302507476/xiaodayang/AudioSummary/SpatialLogic"

# ---------------- 可按需覆盖的训练参数（也可在外部 export） ----------------
# Qwen3.5 多模态指令权重；该目录需包含 config.json、chat_template.jinja、preprocessor_config.json。
MODEL_PATH=${MODEL_PATH:-"/apdcephfs_gy5/share_302507476/xiaodayang/AudioSummary/mkxlsx_train/ckpt/qwen3.5_9B"}
DATA_ROOT=${DATA_ROOT:-"/apdcephfs_gy5/share_302507476/xiaodayang/SpatialLogic_Data"}
TRAIN_FILES=${TRAIN_FILES:-"${SCRIPT_DIR}/training_data/equal/equal_ego_ego4d.json ${SCRIPT_DIR}/training_data/equal/equal_ego_epic.json ${SCRIPT_DIR}/training_data/equal/equal_short_trainset.json ${SCRIPT_DIR}/training_data/leap/all_168042.jsonl ${SCRIPT_DIR}/training_data/short/tag50_finished.json ${SCRIPT_DIR}/training_data/short/cot50fin_cleaned_filtered_translated.json ${SCRIPT_DIR}/training_data/tag_long_ego/all_train.jsonl"}
SYS_PROMPT=${SYS_PROMPT:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"/apdcephfs_gy2_302507476/share_302507476/xiaodayang/AudioSummary/SpatialLogic/ckpt/qwen3.5_spatiallogic"}
LOG_DIR=${LOG_DIR:-"${OUTPUT_DIR}/tb"}
MISSING_DIR=${MISSING_DIR:-"${SCRIPT_DIR}/missing"}
BROKEN_DIR=${BROKEN_DIR:-"${SCRIPT_DIR}/missing/broken"}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-2e-5}
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_STEPS=${MAX_STEPS:-10000}
MAX_SAMPLES=${MAX_SAMPLES:-0}
GRAD_ACC=${GRAD_ACC:-3}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
SKIP_MISSING_IMAGES=${SKIP_MISSING_IMAGES:-1}
FREEZE_VISUAL=${FREEZE_VISUAL:-0}

# 使用已验证支持 Qwen3.5 的环境；不要安装 transformers==4.57.3 覆盖它。
ACTIVATE_CMD=${ACTIVATE_CMD:-"source /apdcephfs_gy2/share_302507476/xiaodayang/miniconda3/bin/activate qwenvl"}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-""}
# ------------------------------------------------------------------------

# 创建配置文件目录（需为共享路径，让所有节点都能读到）
mkdir -p "${SCRIPT_DIR}/accelerate_config_machine"

# 机器数量与总进程数；默认从第一个 node_spec 的 :GPU数 读取。
NUM_MACHINES=${#node_ips[@]}
DEFAULT_GPUS_PER_MACHINE=${node_specs[0]##*:}
GPUS_PER_MACHINE=${GPUS_PER_MACHINE:-$DEFAULT_GPUS_PER_MACHINE}
if ! [[ "$GPUS_PER_MACHINE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid GPUS_PER_MACHINE: $GPUS_PER_MACHINE" >&2
    exit 1
fi
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_MACHINE))
echo "[pdsh.sh] Nodes=${node_ips[*]} NUM_MACHINES=$NUM_MACHINES GPUS_PER_MACHINE=$GPUS_PER_MACHINE NUM_PROCESSES=$NUM_PROCESSES"

# 为每台机器创建配置文件
for i in "${!node_ips[@]}"; do
    config_file="${SCRIPT_DIR}/accelerate_config_machine/${i}.yaml"
    
    cat > $config_file << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: 'all'
machine_rank: $i
main_training_function: main
main_process_ip: $MASTER_ADDR
main_process_port: $MASTER_PORT
mixed_precision: bf16
num_machines: $NUM_MACHINES
num_processes: $NUM_PROCESSES
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
done

# 创建 DeepSpeed 配置文件
ds_config_file="${SCRIPT_DIR}/accelerate_config_machine/ds_config.json"

cat > $ds_config_file << 'EOF'
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 1,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": "auto"
        }
    }
}
EOF

for required_path in "$SCRIPT_DIR/train_cot_only.py" "$MODEL_PATH/config.json"; do
    if [ ! -f "$required_path" ]; then
        echo "Required file not found: $required_path" >&2
        exit 1
    fi
done
if [ ! -d "$DATA_ROOT" ]; then
    echo "Required data root not found: $DATA_ROOT" >&2
    exit 1
fi
for train_file in $TRAIN_FILES; do
    if [ ! -f "$train_file" ]; then
        echo "Required training file not found: $train_file" >&2
        exit 1
    fi
done
if [ -n "$SYS_PROMPT" ] && [ ! -f "$SYS_PROMPT" ]; then
    echo "Required system prompt not found: $SYS_PROMPT" >&2
    exit 1
fi

if [ -n "$RESUME_FROM_CHECKPOINT" ] && [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
    echo "Checkpoint directory not found: $RESUME_FROM_CHECKPOINT" >&2
    exit 1
fi

RESUME_ARG=""
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}"
fi
TRUST_REMOTE_CODE_ARG=""
if [ "$TRUST_REMOTE_CODE" = "1" ]; then
    TRUST_REMOTE_CODE_ARG="--trust_remote_code"
fi
SYS_PROMPT_ARG=""
if [ -n "$SYS_PROMPT" ]; then
    SYS_PROMPT_ARG="--sys_prompt ${SYS_PROMPT}"
fi
SKIP_MISSING_IMAGES_ARG="--skip_missing_images"
if [ "$SKIP_MISSING_IMAGES" = "0" ]; then
    SKIP_MISSING_IMAGES_ARG="--no_skip_missing_images"
fi
FREEZE_VISUAL_ARG=""
if [ "$FREEZE_VISUAL" = "1" ]; then
    FREEZE_VISUAL_ARG="--freeze_visual"
fi
mkdir -p "$MISSING_DIR" "$BROKEN_DIR"

# 在每台机器上并行启动训练
for i in "${!node_ips[@]}"; do
    ip=${node_ips[$i]}
    config_file="${SCRIPT_DIR}/accelerate_config_machine/${i}.yaml"
    
    pdsh -w "$ip" "cd ${SCRIPT_DIR} && \
        ${ACTIVATE_CMD} && \
        export MASTER_ADDR=${MASTER_ADDR} && \
        export MASTER_PORT=${MASTER_PORT} && \
        accelerate launch --config_file ${config_file} --main_process_port ${MASTER_PORT} \
            ${SCRIPT_DIR}/train_cot_only.py \
            --deepspeed ${ds_config_file} \
            --model_path ${MODEL_PATH} \
            --data_root ${DATA_ROOT} \
            --train_files ${TRAIN_FILES} \
            --output_dir ${OUTPUT_DIR} \
            --log_dir ${LOG_DIR} \
            --missing_dir ${MISSING_DIR} \
            --broken_dir ${BROKEN_DIR} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LR} \
            --gradient_accumulation_steps ${GRAD_ACC} \
            --max_length ${MAX_LENGTH} \
            --max_steps ${MAX_STEPS} \
            --max_samples ${MAX_SAMPLES} \
            --gradient_checkpointing \
            ${SYS_PROMPT_ARG} \
            ${SKIP_MISSING_IMAGES_ARG} \
            ${FREEZE_VISUAL_ARG} \
            ${RESUME_ARG} \
            ${TRUST_REMOTE_CODE_ARG}" &
done

# 等待所有训练进程
wait