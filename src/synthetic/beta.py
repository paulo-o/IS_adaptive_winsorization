import numpy as np
import pandas as pd
from tqdm import trange
import plotnine as pn
from argparse import ArgumentParser
import sys
import math

from src.utils.general_utils import get_folder, is_notebook
from src.utils.winsorized_IS_utils import (
    mse,
    mad,
    winsorize,
    bis_threshold,
    cv_threshold,
)
from src.utils.synthetic_utils import beta_proposal_sample as proposal_sample
from src.utils.synthetic_utils import beta_IS_ratio as ratio_IS
from src.utils.synthetic_utils import output_file_exists

if is_notebook():
    # Use default values in notebook
    class Args:
        sim_size = 10000
        n = 1000
        threshold_set = np.sqrt(n) * np.array([0.25, 0.5, 1, 1.5, 2])
        seed = 0
        params_min = 0.8
        params_max = 1
        params_steps = 10
        recreate_if_existing = True

    args = Args()
else:
    # Read parameters from command line
    parser = ArgumentParser()
    parser.add_argument(
        "--sim_size",
        "-ss",
        default=10000,
        type=int,
        help="Number of simulation runs for MSE, MAD calculation.",
    )
    parser.add_argument(
        "--n",
        "-n",
        default=1000,
        type=int,
        help="Number of samples in each simulation.",
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
        "--seed", "-s", default=0, type=int, help="Seed value to use.",
    )
    parser.add_argument(
        "--params_min",
        "-min",
        default=0.8,
        type=float,
        help="Minimum value for parameter.",
    )
    parser.add_argument(
        "--params_max",
        "-max",
        default=1,
        type=float,
        help="Maximum value for parameter.",
    )
    parser.add_argument(
        "--params_steps",
        "-steps",
        default=10,
        type=int,
        help="Number of steps for parameters.",
    )
    parser.add_argument(
        "--recreate_if_existing",
        "-rie",
        action="store_true",
        help="Whether to recreate errors DataFrame if it already exists.",
    )
    args = parser.parse_args()

SIM_SIZE = args.sim_size
N = args.n
WINSORIZATION_THRESHOLD_SET = args.threshold_set
SEED = args.seed
PARAMS_MIN = args.params_min
PARAMS_MAX = args.params_max
PARAMS_STEP = args.params_steps
RECREATE_IF_EXISTING = args.recreate_if_existing

EXAMPLE = "beta"
PARAMS_VECTOR = np.linspace(PARAMS_MIN, PARAMS_MAX, PARAMS_STEP)  # Changing parameters
MEAN_VECTOR = math.pi*np.ones_like(PARAMS_VECTOR)  # Values to be estimated

if output_file_exists(EXAMPLE, args) and not RECREATE_IF_EXISTING:
    sys.exit("DataFrame with errors already exist; skipping.")

errors_df = pd.DataFrame(
    index=PARAMS_VECTOR,
    columns=[["MSE"] * 5 + ["MAD"] * 5,
             ["IS", "BALANCED_IS", "CV_IS", "FIXED", "FITHIAN-WAGER"] * 2],
)

cv_trunc = []
bis_trunc = []
fw_trunc = []

np.random.seed(SEED)
for example_n, parameter in enumerate(PARAMS_VECTOR):
    mean = MEAN_VECTOR[example_n]

    IS = np.zeros((SIM_SIZE))
    CV_IS = np.zeros((SIM_SIZE))
    BALANCED_IS = np.zeros((SIM_SIZE))
    FIXED = np.zeros((SIM_SIZE))
    FITHIAN_WAGER = np.zeros((SIM_SIZE))

    for sim in trange(
        SIM_SIZE, desc=f"Example {example_n+1}/{len(PARAMS_VECTOR)}", file=sys.stdout
    ):

        x = proposal_sample(N, parameter)
        y = np.asarray(ratio_IS(x, parameter))

        # Usual IS
        IS[sim] = np.mean(y)

        # CV IS
        winsorization_value_cv = cv_threshold(
            y, WINSORIZATION_THRESHOLD_SET, number_of_folds=5, verbose=False
        )
        cv_trunc.append(winsorization_value_cv)
        CV_IS[sim] = np.mean(winsorize(y, winsorization_value_cv))

        # Balanced IS
        winsorization_value_bis = bis_threshold(
            y, WINSORIZATION_THRESHOLD_SET, c=(1 + np.sqrt(5)), t=1/np.sqrt(N)
        )
        bis_trunc.append(winsorization_value_bis)
        BALANCED_IS[sim] = np.mean(winsorize(y, winsorization_value_bis))

        # Fixed IS
        FIXED[sim] = np.mean(winsorize(y, np.sqrt(N)))

        # Fithian-Wager IS
        second_largest_value = np.partition(y, -2)[-2]

        FITHIAN_WAGER[sim] = np.mean(winsorize(y, second_largest_value))
        fw_trunc.append(second_largest_value)

    errors_df.loc[parameter]["MSE"] = [
        mse(IS, mean),
        mse(BALANCED_IS, mean),
        mse(CV_IS, mean),
        mse(FIXED, mean),
        mse(FITHIAN_WAGER, mean),
    ]
    errors_df.loc[parameter]["MAD"] = [
        mad(IS, mean),
        mad(BALANCED_IS, mean),
        mad(CV_IS, mean),
        mad(FIXED, mean),
        mad(FITHIAN_WAGER, mean),
    ]

# Save DataFrame with errors
output_filename = (
    f"{EXAMPLE}_{SEED}_{'-'.join(map(str, WINSORIZATION_THRESHOLD_SET))}_"
    f"{PARAMS_MIN}_{PARAMS_MAX}_{PARAMS_STEP}_{N}_{SIM_SIZE}"
)
errors_folder = get_folder("eval/synthetic/errors")
errors_df.to_csv(f"{errors_folder}/{output_filename}.csv")
print(f" -saved: {errors_folder}/{output_filename}.csv")

# Plot MSE
MSE_plot_df = errors_df["MSE"].reset_index().melt("index")
MSE_plot_df = MSE_plot_df.astype({"value": "float64"})

plot_MSE = (
    pn.ggplot(MSE_plot_df)
    + pn.theme_light()
    + pn.theme(legend_key=pn.element_rect(color="white"))
    + pn.geom_point(pn.aes(x="index", y="value", color="variable"))
    + pn.scale_color_manual(
        values=["#00BFC4", "#50388d", "#F8766D", "#fd8c04", "#fecd1a"],
    )
    + pn.labs(x="$\\theta$", y="Mean squared error", color=" ")
    + pn.geom_line(
        pn.aes(x="index", y="value", group="variable", color="variable"),
        linetype="dashed",
    )
)
plot_folder = get_folder("eval/synthetic/plots")
plot_MSE.save(
    f"{plot_folder}/{output_filename}_MSE.pdf", width=7, height=4, verbose=False
)
print(f" -saved: {plot_folder}/{output_filename}_MSE.pdf")


# Plot MAD
MAD_plot_df = errors_df["MAD"].reset_index().melt("index")
MAD_plot_df = MAD_plot_df.astype({"value": "float64"})

plot_MAD = (
    pn.ggplot(MAD_plot_df)
    + pn.theme_light()
    + pn.theme(legend_key=pn.element_rect(color="white"))
    + pn.geom_point(pn.aes(x="index", y="value", color="variable"))
    + pn.scale_color_manual(
        values=["#00BFC4", "#50388d", "#F8766D", "#fd8c04", "#fecd1a"],
    )
    + pn.labs(x="$\\theta$", y="Mean absolute deviation", color=" ")
    + pn.geom_line(
        pn.aes(x="index", y="value", group="variable", color="variable"),
        linetype="dashed",
    )
)

plot_MAD.save(
    f"{plot_folder}/{output_filename}_MAD.pdf", width=7, height=4, verbose=False
)
print(f" -saved: {plot_folder}/{output_filename}_MAD.pdf")
