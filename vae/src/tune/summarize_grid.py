import argparse
import pandas as pd

import src.utils.performance as pf


parser = argparse.ArgumentParser(description='Process input directory')
parser.add_argument('--input_dir', '-i', type=str, help='Path to input directory')
args = parser.parse_args()


if __name__ == "__main__":
    # Do grid analyisis on the input directory
    grid_analysis = pf.analyse_grid_run(args.input_dir, split='test', plot_dir=f"{args.input_dir}/plots")
    # Unpack results
    df: pd.DataFrame = grid_analysis['df']    
    # Sort by f1-score_test and show all columns that have more than 1 unique value
    extra_cols = [c for c in df.columns if df[c].astype(str).nunique() > 1]
    df = df[extra_cols].sort_values('f1-score_test', ascending=False)
    # Save to csv
    df.to_csv(f'{args.input_dir}/summary.csv')
   