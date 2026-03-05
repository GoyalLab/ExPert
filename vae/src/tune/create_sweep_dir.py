import argparse
import yaml
import numpy as np

import src.utils.io as io

argparser = argparse.ArgumentParser(description="Create a sweep directory with N configs sampled from a config template.")
argparser.add_argument("--config_template", "-c", type=str, required=True, help="Path to the config template YAML file.")
argparser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save the sampled configs.")
argparser.add_argument("--num_configs", "-n", type=int, default=100, help="Number of configs to sample.")
argparser.add_argument("--random_samples", action='store_true', help="Whether to sample configs randomly or use a grid search.")
args = argparser.parse_args()


if __name__ == "__main__":
    # I/O
    base_config_p = args.config_template
    output_dir = args.output_dir
    N = args.num_configs
    random_samples = args.random_samples
    # Load the config template
    with open(base_config_p, 'r') as f:
        # Read default config file
        space = yaml.safe_load(f)

    # Setup random generator
    rng = np.random.default_rng(42)
    # Sample N configs from space and create run directories
    search_space, run_df = io.sample_configs(space, space, N=N, base_dir=output_dir, random_samples=random_samples)
