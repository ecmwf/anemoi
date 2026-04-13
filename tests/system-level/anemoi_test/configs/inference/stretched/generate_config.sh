# This script generates configuration files for inference based on a checkpoint.

# Create output directory
mkdir -p $(dirname $OUTPUT_PATH)

# Export checkpoint variable for variable substitution
export CHECKPOINT_PATH=$(find $CHECKPOINT_DIR -name "$CHECKPOINT_FILE")

# Paths to training zarr datasets (used for boundary forcings time series)
export LAM_DATASET=$RESULTS_DIR_DATASETS/cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing.zarr
export GLOBAL_DATASET=$RESULTS_DIR_DATASETS/aifs-ea-an-oper-0001-mars-o96-2017-2017-6h-v8-testing-larsfp.zarr

# Generate a config file for the checkpoint
envsubst < $CONFIG_TEMPLATE > $OUTPUT_PATH
echo "Generated config file: $OUTPUT_PATH using checkpoint: $CHECKPOINT_PATH"
echo "  Checkpoint:     $CHECKPOINT_PATH"
echo "  LAM dataset:    $LAM_DATASET"
echo "  Global dataset: $GLOBAL_DATASET"
