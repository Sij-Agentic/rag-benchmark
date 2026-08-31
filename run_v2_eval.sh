#!/bin/bash
# Quick script to run v2 evaluation with custom ground truth

cd /home/ubuntu/rag-benchmark
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate rag-benchmark

# Copy v2 to default location temporarily
cp data/ground_truth_v2.json data/ground_truth.json

echo "Running Pipeline A on V2 corpus..."
python src/evaluate.py --pipeline A --output results/v2_pipeline_a.csv

echo "Running Pipeline B on V2 corpus..."
python src/evaluate.py --pipeline B --output results/v2_pipeline_b.csv

echo "Running Pipeline C on V2 corpus..."
python src/evaluate.py --pipeline C --output results/v2_pipeline_c.csv

echo "✓ All pipelines complete!"
