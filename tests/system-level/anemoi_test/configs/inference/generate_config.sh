# This script generates configuration files for inference based on model checkpoints.

# Build checkpoint list
FILE_LIST=$(find $CHECKPOINT_DIR -name "$CHECKPOINT_PATTERN" | sort)
echo $FILE_LIST

# Create output directory
mkdir -p $(dirname $OUTPUT_PATH)

# Generate a config file for the checkpoint
export CHECKPOINT=$(echo $FILE_LIST | awk '{print $1}')
envsubst < $CONFIG_TEMPLATE > $OUTPUT_PATH
echo "Generated config file: $OUTPUT_PATH using checkpoint: $CHECKPOINT"
