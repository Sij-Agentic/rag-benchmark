#!/usr/bin/env python
"""Quick analysis of V2 corpus results."""

import pandas as pd
from pathlib import Path

def analyze_results(csv_path: Path):
    """Analyze results from one pipeline."""
    df = pd.read_csv(csv_path)

    pipeline = df['pipeline'].iloc[0]
    print(f"\n{'='*60}")
    print(f"Pipeline {pipeline} Results")
    print(f"{'='*60}")

    # Overall stats
    total = len(df)
    errors = df['error'].notna().sum()
    successful = total - errors

    print(f"\nOverall:")
    print(f"  Total questions: {total}")
    print(f"  Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"  Errors: {errors} ({errors/total*100:.1f}%)")

    # By dataset
    print(f"\nBy Dataset:")
    for dataset in df['source_dataset'].unique():
        subset = df[df['source_dataset'] == dataset]
        subset_errors = subset['error'].notna().sum()
        subset_success = len(subset) - subset_errors
        print(f"  {dataset}: {subset_success}/{len(subset)} successful")

    # Retrieval accuracy
    retrieval_hits = df['retrieval_hit_at_k'].sum()
    print(f"\nRetrieval:")
    print(f"  Hit@5: {retrieval_hits}/{successful} ({retrieval_hits/successful*100:.1f}%)")

    # Timing
    avg_ingest = df[df['ingest_time_s'] > 0]['ingest_time_s'].mean()
    avg_query = df[df['query_time_s'] > 0]['query_time_s'].mean()
    print(f"\nTiming:")
    print(f"  Avg ingest: {avg_ingest:.2f}s")
    print(f"  Avg query: {avg_query:.2f}s")

    # Context length
    avg_context = df[df['context_length'] > 0]['context_length'].mean()
    print(f"\nContext:")
    print(f"  Avg length: {avg_context:.0f} chars")

    return df

def main():
    results_dir = Path("results")

    # Analyze all V2 results
    for csv_file in sorted(results_dir.glob("v2_pipeline_*_fixed.csv")):
        analyze_results(csv_file)

    print(f"\n{'='*60}")
    print("Summary Complete")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
