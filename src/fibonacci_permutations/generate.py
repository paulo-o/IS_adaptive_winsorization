from argparse import ArgumentParser

import numpy as np
from tqdm import trange

from src.utils.general_utils import is_notebook, get_folder
from src.utils.fibonacci_permutations_utils import fib_permutations


if is_notebook():
    # Use default values in notebook
    class Args:
        number_obs = 10000
        number_sims = 1000
        initial_permutation_type = "identity"
        n = 10
        seed = 30
        recreate_if_existing = True
    args = Args()
else:
    # Read parameters from command line
    parser = ArgumentParser()
    parser.add_argument(
        "--number_obs",
        "-no",
        default=1000,
        type=int,
        help="Number of generated observations (i.e., saws) in each simulation.",
    )
    parser.add_argument(
        "--number_sims",
        "-ns",
        default=1000,
        type=int,
        help="Number of simulations in each experiment, for MSE and MAD calculations.",
    )
    parser.add_argument(
        "--initial_permutation_type",
        "-t",
        default="identity",
        choices=["random", "identity"],
        type=str,
        help="How the initial permutation is chosen; identity is [0, 1, 2, .., n-1]; "
             "random picks a random permutation of [n]."
    )
    parser.add_argument(
        "--n",
        "-n",
        default=10,
        type=int,
        help="Investigate Fibonacci permutations of 0, 1, ..., n-1.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=30,
        type=int,
        help="Seed to use in generating experiment data.",
    )
    parser.add_argument(
        "--recreate_if_existing",
        "-rie",
        action="store_true",
        help="Whether to recreate errors DataFrame if it already exists.",
    )
    args = parser.parse_args()


NUMBER_OBS = args.number_obs
NUMBER_SIMS = args.number_sims
TYPE = args.initial_permutation_type
N = args.n
SEED = args.seed
RECREATE_IF_EXISTING = args.recreate_if_existing


lists_of_cum_prod_neighbors = np.zeros((NUMBER_SIMS, NUMBER_OBS))
for i in trange(NUMBER_SIMS):
    (list_of_permutations, list_of_cum_prod_neighbors) = fib_permutations(
        N, seed=SEED * i, reps=NUMBER_OBS, initial_permutation_type=TYPE
    )
    lists_of_cum_prod_neighbors[i] = list_of_cum_prod_neighbors

# Save weights and lengths as matrices
folder_weights = get_folder("data/fibonacci_permutations/weights")

np.save(
    f"{folder_weights}/weights_{TYPE}_{NUMBER_SIMS}_{NUMBER_OBS}_{SEED}_{N}.npy",
    lists_of_cum_prod_neighbors,
)
