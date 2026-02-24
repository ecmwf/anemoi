# This script generates configuration files for inference based on a checkpoint.

# Create output directory
mkdir -p $(dirname $OUTPUT_PATH)

# Export checkpoint variable for variable substitution
# export CHECKPOINT_PATH=$(find $CHECKPOINT_DIR -name "$CHECKPOINT_FILE")

# Supply required datasets
export LAM_DATASET="$RESULTS_DIR_DATASETS/aifs-ea-an-oper-0001-mars-O96-2025-2025-6h-v1-testing.zarr"
export GLOBAL_DATASET="$RESULTS_DIR_DATASETS/aifs-ea-an-oper-0001-mars-o48-2025-2025-6h-v1-testing.zarr"

# Generate a config file for the checkpoint
envsubst < $CONFIG_TEMPLATE > $OUTPUT_PATH
echo "Generated config file: $OUTPUT_PATH using checkpoint from huggingface"
