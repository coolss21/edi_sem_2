import os
import glob
import json
import pandas as pd
import numpy as np
import scipy.stats as ss
import scikit_posthocs as sp
import warnings

warnings.filterwarnings("ignore")

TABLES_DIR = "outputs/tables"

def collect_metrics():
    """Collects ARI metrics from all processed datasets."""
    files = glob.glob(os.path.join(TABLES_DIR, "*_comparison_metrics.csv"))
    if not files:
        print("No comparison metrics found. Run main.py on multiple datasets first.")
        return None

    results = []
    for f in files:
        ds_name = os.path.basename(f).replace("_comparison_metrics.csv", "")
        try:
            df = pd.read_csv(f)
            # We want to extract the ARI for each model
            row_dict = {"dataset": ds_name}
            for _, row in df.iterrows():
                row_dict[row["model_name"]] = row["ARI"]
            results.append(row_dict)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not results:
        return None

    df_results = pd.DataFrame(results).dropna()
    return df_results

def run_nemenyi_test(df_results):
    print("============================================================")
    print("  STATISTICAL SIGNIFICANCE EVALUATION (NEMENYI TEST)")
    print("============================================================")
    
    models = [c for c in df_results.columns if c != "dataset"]
    print(f"Evaluated Datasets: {len(df_results)}")
    print(f"Models Evaluated: {models}\n")

    # Friedman Test
    # df_results[models].values has shape (N_datasets, N_models)
    data = [df_results[m].values for m in models]
    
    stat, p_value = ss.friedmanchisquare(*data)
    print(f"Friedman Test Statistic: {stat:.4f}, p-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("Result: SIGNIFICANT DIFFERENCE exists among the models (p < 0.05).")
        print("\nRunning Nemenyi Post-hoc Test...")
        
        # Scikit-posthocs requires data in a specific format for nemenyi
        # We can pass the DataFrame directly
        nemenyi_p_values = sp.posthoc_nemenyi_friedman(df_results.set_index("dataset")[models].values)
        nemenyi_p_values.columns = models
        nemenyi_p_values.index = models
        
        print("\nNemenyi p-value matrix (pairwise comparisons):")
        print(nemenyi_p_values.round(4))
        
        # Calculate average ranks
        ranks = df_results[models].rank(axis=1, ascending=False)
        avg_ranks = ranks.mean().sort_values()
        
        print("\nAverage Ranks (Lower is better):")
        for m, r in avg_ranks.items():
            print(f"  {m}: {r:.2f}")
            
    else:
        print("Result: NO SIGNIFICANT DIFFERENCE among models (p >= 0.05).")

if __name__ == "__main__":
    df = collect_metrics()
    if df is not None:
        if len(df) < 3:
            print(f"Found {len(df)} datasets. Need at least 3 datasets for a reliable Friedman test.")
            print("Run more datasets using `python main.py --datasets <ds1> <ds2> ...`")
        else:
            run_nemenyi_test(df)
