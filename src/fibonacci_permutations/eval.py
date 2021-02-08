from argparse import ArgumentParser
import numpy as np
import pandas as pd
import plotnine as pn
from tqdm import trange

from src.utils.general_utils import get_folder, is_notebook
from src.utils.fibonacci_permutations_utils import fast_fib
from src.utils.winsorized_IS_utils import (
    mse,
    mad,
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
        threshold_set = np.array(
            [2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12]
        )
        initial_permutation_type = "identity"
        n = 10
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
        default=10,
        type=int,
        help="Number of nodes in grid; edges = nodes - 1.",
    )
    parser.add_argument(
        "--initial_permutation_type",
        "-t",
        default="identity",
        choices=["random", "identity"],
        type=str,
        help="How the initial permutation is chosen; identity is [0, 1, 2, .., n-1]; "
        "random picks a random permutation of [n].",
    )
    parser.add_argument(
        "--recreate_if_existing",
        "-rie",
        action="store_true",
        help="Whether to recreate errors DataFrame if it already exists.",
    )
    args = parser.parse_args()

NUMBER_SIMS = args.number_sims
NUMBER_OBS = args.number_obs
SEED = args.seed
WINSORIZATION_THRESHOLD_SET = np.asarray(args.threshold_set)
N = args.n
TYPE = args.initial_permutation_type
RECREATE_IF_EXISTING = args.recreate_if_existing

mean = fast_fib(N + 1)

errors_df = pd.DataFrame(
    index=[TYPE],
    columns=[
        ["MSE"] * 5 + ["MAD"] * 5,
        ["IS", "BALANCED_IS", "CV_IS", "FIXED", "FITHIAN-WAGER"] * 2,
    ],
)

cv_trunc = []
bis_trunc = []
fw_trunc = []

weights = np.load(
    f"data/fibonacci_permutations/weights/"
    f"weights_{TYPE}_{NUMBER_SIMS}_{NUMBER_OBS}_{SEED}_{N}.npy"
)

IS = np.zeros((NUMBER_SIMS))
CV_IS = np.zeros((NUMBER_SIMS))
BALANCED_IS = np.zeros((NUMBER_SIMS))
FIXED = np.zeros((NUMBER_SIMS))
FITHIAN_WAGER = np.zeros((NUMBER_SIMS))

for sim in trange(NUMBER_SIMS):
    y = weights[sim, :]

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
        y, WINSORIZATION_THRESHOLD_SET, c=(1 + np.sqrt(5)), t=1 / np.sqrt(NUMBER_OBS),
    )
    bis_trunc.append(winsorization_value_bis)
    BALANCED_IS[sim] = np.mean(winsorize(y, winsorization_value_bis))

    # Fixed IS
    FIXED[sim] = np.mean(winsorize(y, np.sqrt(NUMBER_OBS)))

    # Fithian-Wager IS
    second_largest_value = np.partition(y, -2)[-2]

    FITHIAN_WAGER[sim] = np.mean(winsorize(y, second_largest_value))
    fw_trunc.append(second_largest_value)


errors_df.loc[TYPE]["MSE"] = [
    mse(IS, mean),
    mse(BALANCED_IS, mean),
    mse(CV_IS, mean),
    mse(FIXED, mean),
    mse(FITHIAN_WAGER, mean),
]
errors_df.loc[TYPE]["MAD"] = [
    mad(IS, mean),
    mad(BALANCED_IS, mean),
    mad(CV_IS, mean),
    mad(FIXED, mean),
    mad(FITHIAN_WAGER, mean),
]


# Save DataFrame with errors
output_filename = (
    f"{TYPE}_{SEED}_{NUMBER_SIMS}_{NUMBER_OBS}_"
    f"{'-'.join(map(str, WINSORIZATION_THRESHOLD_SET))}_{N}"
)
errors_folder = get_folder("eval/fibonacci_permutations/errors")
errors_df.to_csv(f"{errors_folder}/{output_filename}.csv")
print(f" -saved: {errors_folder}/{output_filename}.csv")


# Plot MSE
MSE_plot_df = errors_df["MSE"].reset_index().melt("index")
MSE_plot_df = MSE_plot_df.astype({"value": "float64"})
plot_MSE = (
    pn.ggplot(MSE_plot_df)
    + pn.theme_light()
    + pn.theme(legend_key=pn.element_rect(color="white"))
    + pn.geom_point(pn.aes(x="variable", y="value"))
    + pn.scale_color_manual(
        values=["#00BFC4", "#50388d", "#F8766D", "#fd8c04", "#fecd1a"],
    )
    + pn.labs(x="estimators", y="Mean squared error", color=" ")
)

plot_folder = get_folder("eval/fibonacci_permutations/plots")
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
    + pn.geom_point(pn.aes(x="variable", y="value"))
    + pn.scale_color_manual(
        values=["#00BFC4", "#50388d", "#F8766D", "#fd8c04", "#fecd1a"],
    )
    + pn.labs(x="estimators", y="Mean absolute deviation", color=" ")
)

plot_MAD.save(
    f"{plot_folder}/{output_filename}_MAD.pdf", width=7, height=4, verbose=False
)
print(f" -saved: {plot_folder}/{output_filename}_MAD.pdf")


# Plot histograms
estimators = {
    "IS": IS,
    "CV_IS": CV_IS,
    "BALANCED_IS": BALANCED_IS,
    "FIXED": FIXED,
    "FITHIAN-WAGER": FITHIAN_WAGER,
}

for estimator, estimates in estimators.items():
    plot = (
        pn.ggplot(pd.DataFrame({"estimates": estimates}), pn.aes(x="estimates"))
        + pn.theme_light()
        + pn.labs(title=f"{estimator} $n={N}$")
        + pn.geom_histogram(bins=30)
        + pn.geom_vline(
            xintercept=float(fast_fib(N + 1)), color="red", alpha=1, linetype="dotted"
        )
    )
    plot.save(
        filename=(
            f"{plot_folder}/histogram_estimates_{estimator}-"
            f"{TYPE}_{SEED}_{NUMBER_SIMS}_{NUMBER_OBS}_"
            f"{'-'.join(map(str, WINSORIZATION_THRESHOLD_SET))}_{N}_{TYPE}.png"
        ),
        height=4,
        width=4,
        units="in",
        dpi=1000,
        verbose=False
    )
