import numpy as np
import pandas as pd
import plotnine as pn

from src.utils.general_utils import get_folder

errors_folder = "eval/synthetic/errors"

params = {
    "beta": {
        "seed": 0,
        "threshold_set": np.sqrt(1000) * np.array([0.25, 0.5, 1, 1.5, 2]),
        "params_min": 0.8,
        "params_max": 1,
        "params_steps": 10,
        "n": 1000,
        "sim_size": 10000,
    },
    "chi_squared": {
        "seed": 10,
        "threshold_set": np.sqrt(1000) * np.array([0.25, 0.5, 1, 1.5, 2]),
        "params_min": 66,
        "params_max": 75,
        "params_steps": 10,
        "n": 1000,
        "sim_size": 10000,
    },
    "exponential": {
        "seed": 1,
        "threshold_set": np.sqrt(1000) * np.array([0.25, 0.5, 1, 1.5, 2]),
        "params_min": 1.25,
        "params_max": 2,
        "params_steps": 10,
        "n": 1000,
        "sim_size": 10000,
    },
    "normal": {
        "seed": 5345,
        "threshold_set": np.sqrt(1000) * np.array([0.25, 0.5, 1, 1.5, 2]),
        "params_min": 0.6,
        "params_max": 1,
        "params_steps": 10,
        "n": 1000,
        "sim_size": 10000,
    },
}

plots = []

for example in params:
    errors_file = (
        f"{example}_{params[example]['seed']}_"
        f"{'-'.join(map(str, params[example]['threshold_set']))}_"
        f"{params[example]['params_min']}_{params[example]['params_max']}_"
        f"{params[example]['params_steps']}_{params[example]['n']}_"
        f"{params[example]['sim_size']}.csv"
    )

    errors_df = pd.read_csv(
        f"{errors_folder}/{errors_file}", index_col=[0], header=[0, 1]
    )

    MSE_plot_df = errors_df["MSE"].reset_index().melt("index")
    MSE_plot_df = MSE_plot_df.astype({"value": "float64"})
    MSE_plot_df = MSE_plot_df.drop(
        MSE_plot_df[(MSE_plot_df.variable == "FITHIAN-WAGER")].index
    )

    plot_MSE = (
        pn.ggplot(MSE_plot_df)
        + pn.theme_light()
        + pn.theme(legend_key=pn.element_rect(color="white"))
        + pn.labs(x="$\\theta$", y="Mean squared error", color=" ")
        + pn.geom_line(
            pn.aes(x="index", y="value", linetype="variable", color="variable"), size=1
        )
        + pn.scales.scale_linetype_manual(
            ["solid", "dotted", "dashdot", "--"], guide=False
        )
        + pn.scales.scale_color_manual(
            values=["#F8766D", "#50388d", "#fd8c04", "#fecd1a"],
        )
        + pn.theme(
            text=pn.element_text(family="times", size=18), legend_position="none"
        )
    )

    MAD_plot_df = errors_df["MAD"].reset_index().melt("index")
    MAD_plot_df = MAD_plot_df.astype({"value": "float64"})
    MAD_plot_df = MAD_plot_df.drop(
        MAD_plot_df[(MAD_plot_df.variable == "FITHIAN-WAGER")].index
    )

    plot_MAD = (
        pn.ggplot(MAD_plot_df)
        + pn.theme_light()
        + pn.theme(legend_key=pn.element_rect(color="white"))
        + pn.labs(x="$\\theta$", y="Mean absolute deviation", color=" ")
        + pn.geom_line(
            pn.aes(x="index", y="value", linetype="variable", color="variable"), size=1
        )
        + pn.scales.scale_linetype_manual(
            ["solid", "dotted", "dashdot", "--"], guide=False
        )
        + pn.scales.scale_color_manual(
            values=["#F8766D", "#50388d", "#fd8c04", "#fecd1a"],
        )
        + pn.theme(
            text=pn.element_text(family="times", size=18), legend_position="none"
        )
    )

    plot_folder = get_folder("submit")
    plot_MSE.save(
        f"{plot_folder}/{example}_MSE.png", width=7, height=4.8, verbose=False, dpi=300
    )
    plot_folder = get_folder("submit")
    plot_MAD.save(
        f"{plot_folder}/{example}_MAD.png", width=7, height=4.8, verbose=False, dpi=300
    )
