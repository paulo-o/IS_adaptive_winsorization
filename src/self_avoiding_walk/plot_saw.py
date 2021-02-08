from argparse import ArgumentParser
from tqdm import tqdm
import random

from src.utils.general_utils import is_notebook
from src.utils.self_avoiding_walk_utils import run_SAW, plot_grid, print_diagnostics


if is_notebook():
    # Use default values in notebook
    class Args:
        n = 11
        type = "interior_traps"
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 23, 27, 28, 29, 31, 36, 43]
        verbose = False

    args = Args()
else:
    # Read parameters from command line
    parser = ArgumentParser()
    parser.add_argument(
        "--n",
        "-n",
        default=11,
        type=int,
        help="Number of nodes in grid; edges = nodes - 1.",
    )
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
        "--seeds",
        "-s",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 23, 27, 28, 29, 31, 36, 43],
        type=int,
        nargs="+",
        help="Set of seeds to use when generating saw plots.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        type=bool,
        help="Whether to print diagnostics information on the generated saw.",
    )
    args = parser.parse_args()

N = args.n
TYPE = args.type
SEEDS = args.seeds
VERBOSE = args.verbose


for seed in tqdm(SEEDS):
    random.seed(seed)
    moves, log_probabilities, is_successful = run_SAW(N, TYPE, verbose=VERBOSE)

    # Adjust moves for plotting
    moves = [list(elem) for elem in moves]
    for i in range(len(moves)):
        moves[i][0] = N - moves[i][0]
        moves[i][1] = moves[i][1] - 1

    plot_grid(moves, N, seed, TYPE, is_successful, save_file=True)
    if VERBOSE:
        print_diagnostics(N, moves, log_probabilities, is_successful)
