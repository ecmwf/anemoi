# This script generates configuration files for inference based on a checkpoint.

# Create output directory
mkdir -p $(dirname $OUTPUT_PATH)

# Export checkpoint variable for variable substitution
export CHECKPOINT_PATH=$(find $CHECKPOINT_DIR -name "$CHECKPOINT_FILE")

# Generate a config file for the checkpoint
envsubst < $CONFIG_TEMPLATE > $OUTPUT_PATH
echo "Generated config file: $OUTPUT_PATH using checkpoint: $CHECKPOINT_PATH"
