#!/bin/bash

# Directories
export SCRATCH_DIR="/marimo/jade_data"
export FINAL_DIR="/marimo/data/corpus"

mkdir -p $SCRATCH_DIR
mkdir -p $FINAL_DIR

DATASET="UCI-HAR"
REGIMES=("r1" "r2" "r3" "r4" "r5" "r6")
SUBJECTS=$(seq 1 30)

echo "Starting Local Sequential Generation..."

# Preprocess Data to fast local scratch
python scripts/prepare_real_data.py --dataset $DATASET --output_dir $SCRATCH_DIR

for REGIME in "${REGIMES[@]}"; do
    for SUBJECT_IDX in $SUBJECTS; do
        
        if [ "$REGIME" = "r6" ]; then
            SEVERITY_ARG="--severity mid"
        else
            SEVERITY_ARG=""
        fi

        echo "Running: Dataset=$DATASET | Regime=$REGIME | Subj=$SUBJECT_IDX"
        
        python scripts/generate_corpus.py \
            --dataset $DATASET \
            --regime $REGIME \
            --subject_idx $SUBJECT_IDX \
            --output_dir $SCRATCH_DIR \
            $SEVERITY_ARG

    done
done

echo "Copying Parquet chunks to home directory..."
rsync -av $SCRATCH_DIR/corpus/$DATASET/ $FINAL_DIR/$DATASET/

echo "Local Generation completed successfully."
