#!/bin/bash
# Eagle3 训练启动脚本（支持 online / offline 模式，本地 Qwen3-8B）

set -euo pipefail

# 自动激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

cd "${PROJECT_DIR}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dt) DT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) shift ;;
    esac
done

DT="${DT:-a800}"
MODE="${MODE:-online}"
DATASET="${DATASET:-nemotron}"

if [[ "${DT}" != "qz" && "${DT}" != "a800" && "${DT}" != "h100" ]]; then
    echo "错误: --dt 须为 qz、a800 或 h100"
    exit 1
fi

if [[ "${MODE}" != "online" && "${MODE}" != "offline" ]]; then
    echo "错误: --mode 须为 online 或 offline"
    exit 1
fi

# ========================================
# 分布式环境
# ========================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [ -n "${PET_NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
fi
NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29503}}"

if [ "${NNODES}" -gt 1 ] 2>/dev/null && { [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; }; then
    echo "错误: 多机训练 (NNODES=${NNODES}) 须设置 MASTER_ADDR 或 PET_MASTER_ADDR 为可互通的主节点地址。" >&2
    exit 1
fi

export MASTER_ADDR
export MASTER_PORT

# ========================================
# 模型与训练参数
# ========================================
TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-${PROJECT_DIR}/models}"
if [ "${DT}" = "qz" ]; then
    export WANDB_MODE=offline
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
elif [ "${DT}" = "h100" ]; then
    if [ -n "${WHZ_DIR:-}" ]; then
        TARGET_MODEL="${TARGET_MODEL:-${WHZ_DIR}/models/Qwen/Qwen3-8B}"
    else
        TARGET_MODEL="${TARGET_MODEL:-${LOCAL_MODEL_ROOT}/Qwen/Qwen3-8B}"
    fi
else
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi

NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
WARMUP_RATIO="${WARMUP_RATIO:-0.015}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
TTT_LENGTH="${TTT_LENGTH:-7}"
DRAFT_ACCUMULATION_STEPS="${DRAFT_ACCUMULATION_STEPS:-1}"
if [ "${DT}" = "a800" ] && [ -z "${ENABLE_THINKING+x}" ]; then
    ENABLE_THINKING="on"
else
    ENABLE_THINKING="${ENABLE_THINKING:-off}"
fi

# ========================================
# 数据集配置
# 默认对齐 FlashMTP 的 Nemotron regen 数据；也支持手动覆盖
# - 训练直接依赖 TRAIN_DATA_PATH，不再在启动脚本内做采样
# ========================================
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-40000}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen3-thinking}"
IS_PREFORMATTED="${IS_PREFORMATTED:-}"
TRAIN_ONLY_LAST_TURN="${TRAIN_ONLY_LAST_TURN:-}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
TRAIN_HIDDEN_STATES_PATH="${TRAIN_HIDDEN_STATES_PATH:-}"
EVAL_HIDDEN_STATES_PATH="${EVAL_HIDDEN_STATES_PATH:-}"
if [ "${DT}" = "qz" ]; then
    DEFAULT_TRAIN_DATA_PATH="/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl"
elif [ "${DT}" = "h100" ]; then
    DEFAULT_TRAIN_DATA_PATH="../training_data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl"
else
    DEFAULT_TRAIN_DATA_PATH="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
fi

CACHE_DIR="${CACHE_DIR:-./cache/data/eagle3_${DATASET}_${DATA_NUM_SAMPLES}}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DEFAULT_TRAIN_DATA_PATH}}"

# ========================================
# Eagle3 模型参数
# ========================================
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
TP_SIZE="${TP_SIZE:-1}"
SP_ULYSSES_SIZE="${SP_ULYSSES_SIZE:-1}"
SP_RING_SIZE="${SP_RING_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-20}"
DRAFT_MODEL_CONFIG="${DRAFT_MODEL_CONFIG:-}"
EMBEDDING_KEY="${EMBEDDING_KEY:-model.embed_tokens.weight}"
LM_HEAD_KEY="${LM_HEAD_KEY:-lm_head.weight}"
MODEL_DOWNLOAD_DIR="${MODEL_DOWNLOAD_DIR:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-}"

# ========================================
# 日志与保存
# ========================================
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-eagle3-training}"
WANDB_NAME="${WANDB_NAME:-eagle3_${DT}_${DATASET}_n${DATA_NUM_SAMPLES}_ep${NUM_EPOCHS}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-eagle3_${DT}_${DATASET}_n${DATA_NUM_SAMPLES}}"
WANDB_KEY="${WANDB_KEY:-}"

RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"

OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/eagle3_${DT}_${DATASET}_${MODE}_think_${ENABLE_THINKING}_n${DATA_NUM_SAMPLES}_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"

# ========================================
# 准备训练数据
# - 直接使用 TRAIN_DATA_PATH（默认优先走 FlashMTP 同款 Nemotron regen 数据）
# ========================================
mkdir -p "${CACHE_DIR}"

if [ ! -f "${TRAIN_DATA_PATH}" ]; then
    echo "错误: TRAIN_DATA_PATH 不存在: ${TRAIN_DATA_PATH}"
    echo "请手动设置 TRAIN_DATA_PATH，并确保它是可直接训练的 jsonl 文件。"
    exit 1
fi

if [ "${MODE}" = "offline" ]; then
    if [ -z "${TRAIN_HIDDEN_STATES_PATH}" ]; then
        echo "错误: offline 模式必须设置 TRAIN_HIDDEN_STATES_PATH"
        exit 1
    fi
    if [ ! -f "${TRAIN_HIDDEN_STATES_PATH}" ]; then
        echo "错误: offline 训练 hidden states 文件不存在: ${TRAIN_HIDDEN_STATES_PATH}"
        exit 1
    fi
fi

if [ -n "${EVAL_DATA_PATH}" ] && [ -n "${EVAL_HIDDEN_STATES_PATH}" ]; then
    echo "错误: EVAL_DATA_PATH 与 EVAL_HIDDEN_STATES_PATH 不能同时设置"
    exit 1
fi

# ========================================
# 显示配置
# ========================================
echo "=========================================="
echo "Eagle3 训练启动脚本"
echo "=========================================="
echo "运行环境: --dt ${DT} (qz | a800 | h100)"
echo "训练模式: ${MODE}"
echo "数据集: ${DATASET}"
echo "------------------------------------------"
echo "目标模型: ${TARGET_MODEL}"
echo "目标模型后端: ${TARGET_MODEL_BACKEND}"
echo "思考模式: ${ENABLE_THINKING}"
echo "训练数据路径: ${TRAIN_DATA_PATH}"
echo "样本规模标识: ${DATA_NUM_SAMPLES}"
if [ "${MODE}" = "offline" ]; then
    echo "训练 hidden states: ${TRAIN_HIDDEN_STATES_PATH}"
    echo "评估 hidden states: ${EVAL_HIDDEN_STATES_PATH:-无}"
else
    echo "评估数据: ${EVAL_DATA_PATH:-无}"
fi
echo "输出目录: ${OUTPUT_DIR}"
echo "缓存目录: ${CACHE_DIR}"
echo "------------------------------------------"
echo "训练配置:"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  批大小: ${BATCH_SIZE}"
echo "  学习率: ${LEARNING_RATE}"
echo "  最大长度: ${MAX_LENGTH}"
echo "  TTT长度: ${TTT_LENGTH}"
echo "  Draft accumulation: ${DRAFT_ACCUMULATION_STEPS}"
echo "------------------------------------------"
echo "分布式配置:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NNODES: ${NNODES}"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  TP_SIZE: ${TP_SIZE}"
echo "  SP_ULYSSES_SIZE: ${SP_ULYSSES_SIZE}"
echo "  SP_RING_SIZE: ${SP_RING_SIZE}"
echo "------------------------------------------"
echo "Tracker: ${REPORT_TO}"
echo "=========================================="
echo ""

# 如果输出目录已存在，自动添加数字后缀
original_output_dir="${OUTPUT_DIR}"
suffix=1
while [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; do
    OUTPUT_DIR="${original_output_dir}_${suffix}"
    suffix=$((suffix + 1))
done
if [ "${OUTPUT_DIR}" != "${original_output_dir}" ]; then
    echo "警告: 输出目录 ${original_output_dir} 已存在且非空，自动切换到: ${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

LAUNCHER=(
    torchrun
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

OPTIONAL_ARGS=()

if [ -n "${DRAFT_MODEL_CONFIG}" ]; then
    OPTIONAL_ARGS+=("--draft-model-config" "${DRAFT_MODEL_CONFIG}")
fi

if [ -n "${EVAL_DATA_PATH}" ]; then
    OPTIONAL_ARGS+=("--eval-data-path" "${EVAL_DATA_PATH}")
fi

if [ -n "${TRAIN_HIDDEN_STATES_PATH}" ]; then
    OPTIONAL_ARGS+=("--train-hidden-states-path" "${TRAIN_HIDDEN_STATES_PATH}")
fi

if [ -n "${EVAL_HIDDEN_STATES_PATH}" ]; then
    OPTIONAL_ARGS+=("--eval-hidden-states-path" "${EVAL_HIDDEN_STATES_PATH}")
fi

if [ -n "${IS_PREFORMATTED}" ]; then
    OPTIONAL_ARGS+=("--is-preformatted")
fi

if [ -n "${TRAIN_ONLY_LAST_TURN}" ]; then
    OPTIONAL_ARGS+=("--train-only-last-turn")
fi

if [ -n "${RESUME}" ]; then
    OPTIONAL_ARGS+=("--resume")
fi

if [ -n "${CKPT_DIR}" ]; then
    OPTIONAL_ARGS+=("--ckpt-dir" "${CKPT_DIR}")
fi

if [ -n "${MODEL_DOWNLOAD_DIR}" ]; then
    OPTIONAL_ARGS+=("--model-download-dir" "${MODEL_DOWNLOAD_DIR}")
fi

if [ -n "${TRUST_REMOTE_CODE}" ]; then
    OPTIONAL_ARGS+=("--trust-remote-code")
fi

if [ "${REPORT_TO}" != "none" ]; then
    OPTIONAL_ARGS+=("--report-to" "${REPORT_TO}")
    if [ "${REPORT_TO}" = "wandb" ] && [ -n "${WANDB_PROJECT}" ]; then
        OPTIONAL_ARGS+=("--wandb-project" "${WANDB_PROJECT}")
    fi
    if [ -n "${WANDB_NAME}" ]; then
        OPTIONAL_ARGS+=("--wandb-name" "${WANDB_NAME}")
    fi
    if [ -n "${WANDB_RUN_ID}" ]; then
        OPTIONAL_ARGS+=("--wandb-run-id" "${WANDB_RUN_ID}")
    fi
    if [ -n "${WANDB_KEY}" ]; then
        OPTIONAL_ARGS+=("--wandb-key" "${WANDB_KEY}")
    fi
fi

echo "==> 开始训练 Eagle3"
echo ""

"${LAUNCHER[@]}" ./scripts/train_eagle3.py \
    --target-model-path "${TARGET_MODEL}" \
    --target-model-backend "${TARGET_MODEL_BACKEND}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --embedding-key "${EMBEDDING_KEY}" \
    --lm-head-key "${LM_HEAD_KEY}" \
    --chat-template "${CHAT_TEMPLATE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --max-length "${MAX_LENGTH}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --ttt-length "${TTT_LENGTH}" \
    --draft-accumulation-steps "${DRAFT_ACCUMULATION_STEPS}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
    --build-dataset-num-proc "${BUILD_DATASET_NUM_PROC}" \
    --tp-size "${TP_SIZE}" \
    --sp-ulysses-size "${SP_ULYSSES_SIZE}" \
    --sp-ring-size "${SP_RING_SIZE}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --dist-timeout "${DIST_TIMEOUT}" \
    --seed "${SEED}" \
    "${OPTIONAL_ARGS[@]}"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: ${OUTPUT_DIR}"
echo ""
echo "使用示例："
echo "  from specforge import AutoEagle3DraftModel"
echo "  draft_model = AutoEagle3DraftModel.from_pretrained('${OUTPUT_DIR}/epoch_<epoch>_step_<step>')"
echo "=========================================="
