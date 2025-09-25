#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <stage> <config_file>"
    echo "  stage: data, prepare, embedding, preprocess, or model"
    echo "  config_file: path to configuration file"
    exit 1
}

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    usage
fi

STAGE=$1
CONFIG_FILE=$2

case $STAGE in
    "data")
        echo "Running data stage with config $CONFIG_FILE"
        python data.py --config "$CONFIG_FILE"
        ;;
    "prepare")
        echo "Running prepare stage with config $CONFIG_FILE"
        python prepare.py --config "$CONFIG_FILE" --binary ./prepare.o
        ;;
    "embedding")
        echo "Running embedding stage with config $CONFIG_FILE"
        python embedding.py --config "$CONFIG_FILE"
        ;;
    "preprocess")
        echo "Running preprocess stage with config $CONFIG_FILE"
        echo "Processing train partition..."
        python preprocess.py --config "$CONFIG_FILE" --binary ./preprocess.o --partition train
        echo "Processing valid partition..."
        python preprocess.py --config "$CONFIG_FILE" --binary ./preprocess.o --partition valid
        echo "Processing test partition..."
        python preprocess.py --config "$CONFIG_FILE" --binary ./preprocess.o --partition test
        ;;
    "model")
        echo "Running model stage with config $CONFIG_FILE"
        python model.py --config "$CONFIG_FILE"
        ;;
    *)
        echo "Error: Unknown stage '$STAGE'"
        echo "Available stages: data, prepare, embedding, preprocess, model"
        exit 1
        ;;
esac

exit 0