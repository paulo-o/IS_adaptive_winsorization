from argparse import ArgumentParser
import numpy as np
import random
import sys
from tqdm import trange

from src.utils.general_utils import is_notebook, get_folder
from src.utils.self_avoiding_walk_utils import run_SAW, output_files_exist


if is_notebook():
    # Use default values in notebook
    class Args:
        type = "interior_traps"
        number_sims = 1000
        number_obs = 100
        seed = 30
        n = 11
        verbose = False
        recreate_if_existing = True
    args = Args()
else:
    # Read parameters from command line
    parser = ArgumentParser()
    parser.add_argument(
        "--type",
        "-t",
        default="interior_traps",
        choices=["all_traps", "interior_traps", "no_traps"],
        type=str,
        help="Type of self avoiding walk (saw): all_traps generates saws that can fail "
        "by self-intersecting anywhere; interior_traps generates saws that can only "
        "by self-intersecting in the interior of the grid; and no_traps generates "
        "saws that never self-intersect.",
    )
    parser.add_argument(
        "--number_sims",
        "-ns",
        default=1000,
        type=int,
        help="Number of simulations in each experiment, for MSE and MAD calculations.",
    )
    parser.add_argument(
        "--number_obs",
        "-no",
        default=10000,
        type=int,
        help="Number of generated observations (i.e., saws) in each simulation.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=30,
        type=int,
        help="Seed to use in generating experiment data.",
    )
    parser.add_argument(
        "--n",
        "-n",
        default=11,
        type=int,
        help="Number of nodes in grid; edges = nodes - 1.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        type=bool,
        help="Whether to print which traps have been avoided in generating each saw.",
    )
    parser.add_argument(
        "--recreate_if_existing",
        "-rie",
        action="store_true",
        help="Whether to recreate errors DataFrame if it already exists.",
    )
    args = parser.parse_args()

TYPE = args.type
NUMBER_SIMS = args.number_sims
NUMBER_OBS = args.number_obs
SEED = args.seed
N = args.n
VERBOSE = args.verbose
RECREATE_IF_EXISTING = args.recreate_if_existing

if output_files_exist(args, file="generate") and not RECREATE_IF_EXISTING:
    sys.exit("SAW already exist; skipping.")

random.seed(SEED)
# Create weights and lengths matrices; rows index simulation run and
# column index the observations within each simulation.
weights = np.zeros((NUMBER_SIMS, NUMBER_OBS))
# lengths = np.zeros((NUMBER_SIMS, NUMBER_OBS))

for i in range(NUMBER_SIMS):
    for j in trange(
        NUMBER_OBS, desc=f"Simulation {i+1}/{NUMBER_SIMS}", file=sys.stdout
    ):
        moves, log_probabilities, is_successful = run_SAW(N, TYPE, VERBOSE)
        if is_successful:
            weights[i, j] = np.exp(-np.sum(log_probabilities))
            # lengths[i, j] = len(log_probabilities)

print(f"\nOverall weight mean: {np.mean(np.array(weights).flatten())}")

# Save weights and lengths as matrices
folder_weights = get_folder("data/self_avoiding_walk/weights")
# folder_lengths = get_folder("data/self_avoiding_walk/lengths")

np.save(
    f"{folder_weights}/weights_{TYPE}_{NUMBER_SIMS}_{NUMBER_OBS}_{SEED}_{N}.npy",
    weights,
)
# np.save(
#     f"{folder_lengths}/lengths_{TYPE}_{NUMBER_SIMS}_{NUMBER_OBS}_{SEED}_{N}.npy",
#     lengths,
# )
