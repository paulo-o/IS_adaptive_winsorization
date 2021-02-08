from argparse import ArgumentParser
import numpy as np
import pandas as pd
import plotnine as pn
import sys
from tqdm import trange

from src.utils.general_utils import get_folder, is_notebook
from src.utils.self_avoiding_walk_utils import get_number_of_saws, output_files_exist
from src.utils.winsorized_IS_utils import (
    winsorize,
    bis_threshold,
    cv_threshold,
)

if is_notebook():
    # Use default values in notebook
    class Args:
        number_sims = 1000
        number_obs = 10000
        seed = 30
        threshold_set = [1e21, 5e23, 1e25, 5e26, 1e28]
        n = 11
        verbose = False
        recreate_if_existing = True

    args = Args()
else:
    # Read parameters from command line
    parser = ArgumentParser()
    parser.add_argument(
        "--number_sims",
        "-ns",
        default=100,
        type=int,
        help="Number of simulations in each experiment, for MSE and MAD calculations.",
    )
    parser.add_argument(
        "--number_obs",
        "-no",
        default=100,
        type=int,
        help="Number of generated observations (i.e., saws) in each simulation.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=0,
        type=int,
        help="Seed to use in generating experiment data.",
    )
    parser.add_argument(
        "--threshold_set",
        "-ts",
        default=np.sqrt(1000) * np.array([0.25, 0.5, 1, 1.5, 2]),
        type=int,
        nargs="+",
        help="Thresholds to be considered for winsorization.",
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

# TYPE = args.type
NUMBER_SIMS = args.number_sims
NUMBER_OBS = args.number_obs
SEED = args.seed
WINSORIZATION_THRESHOLD_SET = np.asarray(args.threshold_set)
N = args.n
VERBOSE = args.verbose
RECREATE_IF_EXISTING = args.recreate_if_existing

types = ["all_traps", "interior_traps", "no_traps"]
mean = get_number_of_saws(N)

if output_files_exist(args, file="eval") and not RECREATE_IF_EXISTING:
    sys.exit("DataFrame with errors already exist; skipping.")

errors_df = pd.DataFrame(
    index=types,
    columns=[
        ["MSE"] * 5 + ["MAD"] * 5,
        ["IS", "BALANCED_IS", "CV_IS", "FIXED", "FITHIAN-WAGER"] * 2,
    ],
)

cv_trunc = []
bis_trunc = []
fw_trunc = []

IS = np.zeros((NUMBER_SIMS, 3))
CV_IS = np.zeros((NUMBER_SIMS, 3))
BALANCED_IS = np.zeros((NUMBER_SIMS, 3))
FIXED = np.zeros((NUMBER_SIMS, 3))
FITHIAN_WAGER = np.zeros((NUMBER_SIMS, 3))

np.random.seed(SEED)
for i, type in enumerate(types):
    weights = np.load(
        f"data/self_avoiding_walk/weights/"
        f"weights_{type}_{NUMBER_SIMS}_{NUMBER_OBS}_{SEED}_{N}.npy"
    )

    for sim in trange(NUMBER_SIMS, desc=f"Type {type}", file=sys.stdout):
        y = weights[sim, :]

        # Usual IS
        IS[sim, i] = np.mean(y)

        # CV IS
        winsorization_value_cv = cv_threshold(
            y, WINSORIZATION_THRESHOLD_SET, number_of_folds=5, verbose=False
        )
        cv_trunc.append(winsorization_value_cv)
        CV_IS[sim, i] = np.mean(winsorize(y, winsorization_value_cv))

        # Balanced IS
        winsorization_value_bis = bis_threshold(
            y,
            WINSORIZATION_THRESHOLD_SET,
            c=(1 + np.sqrt(5)),
            t=1 / np.sqrt(NUMBER_OBS),
        )
        bis_trunc.append(winsorization_value_bis)
        BALANCED_IS[sim, i] = np.mean(winsorize(y, winsorization_value_bis))

        # Fixed IS
        FIXED[sim, i] = np.mean(winsorize(y, 1e25))

        # Fithian-Wager IS
        second_largest_value = np.partition(y, -2)[-2]

        FITHIAN_WAGER[sim, i] = np.mean(winsorize(y, second_largest_value))
        fw_trunc.append(second_largest_value)

for i, type in enumerate(types):
    plot_df = pd.DataFrame(columns=["IS", "Balanced IS", "CV IS", "Fixed IS"])
    plot_df["IS"] = IS[:, i]
    plot_df["Balanced IS"] = BALANCED_IS[:, i]
    plot_df["CV IS"] = CV_IS[:, i]
    plot_df["Fixed IS"] = FIXED[:, i]

    plot_df = plot_df.melt()
    plot_df = plot_df.astype({"value": "float64"})

    plot_predictions = (
        pn.ggplot(
            plot_df, pn.aes("variable", "value", color="variable", fill="variable")
        )
        + pn.geom_violin(alpha=0.5)
        + pn.scale_y_log10()
        + pn.geom_hline(yintercept=float(mean), color="red", linetype="dashed")
        + pn.labs(x="estimator", y="prediction (log)")
        + pn.scales.scale_color_manual(
            values=["#F8766D", "#50388d", "#fd8c04", "#fecd1a"],
        )
        + pn.scales.scale_fill_manual(
            values=["#F8766D", "#50388d", "#fd8c04", "#fecd1a"]
        )
        + pn.theme_bw()
        + pn.theme(
            text=pn.element_text(family="times", size=14), legend_position="none"
        )
    )

    plot_folder = get_folder("submit")
    plot_predictions.save(
        f"{plot_folder}/self_avoiding_walk-predictions-{type}-violin.png",
        width=5,
        height=5,
        verbose=False,
        dpi=300,
    )
