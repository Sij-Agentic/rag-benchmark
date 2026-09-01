#!/bin/bash
# V3 Evaluation - Sequential Pipeline Execution
# 71 questions across 4 datasets
#
# CRITICAL: Run pipelines ONE AT A TIME (lesson from V2 OOM issue)

set -e  # Exit on error

echo "================================================================================"
echo "V3 EVALUATION - 71 Questions - Sequential Execution"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

cd /home/ubuntu/rag-benchmark
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate rag-benchmark

# Create logs directory
mkdir -p logs

# Helper function to clear GPU
clear_gpu() {
    echo "Clearing GPU memory..."
    python -c "import torch; torch.cuda.empty_cache(); print('✓ GPU cache cleared')" 2>/dev/null || true
    sleep 5
}

# Helper function to check for errors
check_errors() {
    local result_file=$1
    local pipeline_name=$2

    if [ -f "$result_file" ]; then
        local total=$(wc -l < "$result_file")
        local error_count=$(grep -c "ERROR" "$result_file" 2>/dev/null || echo "0")
        local oom_count=$(grep -c "CUDA out of memory" "$result_file" 2>/dev/null || echo "0")

        echo "  Total rows: $total"
        echo "  Errors: $error_count"
        if [ "$oom_count" -gt "0" ]; then
            echo "  ⚠ WARNING: $oom_count OOM errors detected!"
        fi
    fi
}

echo "================================================================================"
echo "PIPELINE A - Naive Baseline"
echo "================================================================================"
echo "Expected: ~30-40/71 questions (image-only PDFs skipped)"
echo ""

python src/evaluate.py \
    --pipeline A \
    --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_a.csv

echo ""
echo "✓ Pipeline A complete!"
check_errors "results/v3_pipeline_a.csv" "Pipeline A"
echo ""

clear_gpu
echo ""

echo "================================================================================"
echo "PIPELINE B - LlamaParse"
echo "================================================================================"
echo "Expected: 71/71 questions (OCR handles all PDFs)"
echo "Estimated time: 2-3 hours"
echo ""

python src/evaluate.py \
    --pipeline B \
    --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_b.csv

echo ""
echo "✓ Pipeline B complete!"
check_errors "results/v3_pipeline_b.csv" "Pipeline B"
echo ""

clear_gpu
echo ""

echo "================================================================================"
echo "PIPELINE C - VLM"
echo "================================================================================"
echo "Expected: 71/71 questions (VLM handles all PDFs)"
echo "Estimated time: 2-3 hours"
echo ""

python src/evaluate.py \
    --pipeline C \
    --ground-truth data/ground_truth_v3.json \
    --output results/v3_pipeline_c.csv

echo ""
echo "✓ Pipeline C complete!"
check_errors "results/v3_pipeline_c.csv" "Pipeline C"
echo ""

clear_gpu

echo "================================================================================"
echo "ALL PIPELINES COMPLETE!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - results/v3_pipeline_a.csv"
echo "  - results/v3_pipeline_b.csv"
echo "  - results/v3_pipeline_c.csv"
echo ""
echo "Next step: Run LLM-as-judge evaluation"
echo "  python src/llm_judge_v3.py"
echo ""
