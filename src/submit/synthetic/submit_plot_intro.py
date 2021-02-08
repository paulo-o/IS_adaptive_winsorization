import numpy as np
import pandas as pd
import plotnine as pn

from src.utils.general_utils import get_folder


seed = 1
params_min = 1.25
params_max = 2
params_steps = 10
n = 1000
sim_size = 10000
threshold_set = np.sqrt(n) * np.array([0.25, 0.5, 1, 1.5, 2])


errors_folder = "eval/synthetic/errors"
errors_file = (
    f"exponential_{seed}_{'-'.join(map(str, threshold_set))}_"
    f"{params_min}_{params_max}_{params_steps}_{n}_{sim_size}.csv"
)

errors_df = pd.read_csv(f"{errors_folder}/{errors_file}", index_col=[0], header=[0, 1])

MSE_plot_df = errors_df["MSE"].reset_index().melt("index")
MSE_plot_df = MSE_plot_df.astype({"value": "float64"})

MSE_plot_df = MSE_plot_df[MSE_plot_df.variable.isin(["IS", "BALANCED_IS"])]

plot_MSE = (
    pn.ggplot(MSE_plot_df)
    + pn.theme_light()
    + pn.theme(legend_key=pn.element_rect(color="white"))
    + pn.labs(x="$\\theta$", y="Mean squared error", color=" ")
    + pn.geom_line(
        pn.aes(x="index", y="value", linetype="variable", color="variable"), size=1
    )
    + pn.scales.scale_linetype_manual(["solid", "--"], guide=False)
    + pn.scales.scale_color_manual(values=["#F8766D", "#fecd1a"],)
    + pn.theme(text=pn.element_text(family="times", size=18), legend_position="none")
)

MAD_plot_df = errors_df["MAD"].reset_index().melt("index")
MAD_plot_df = MAD_plot_df.astype({"value": "float64"})
MAD_plot_df = MAD_plot_df[MAD_plot_df.variable.isin(["IS", "BALANCED_IS"])]

plot_MAD = (
    pn.ggplot(MAD_plot_df)
    + pn.theme_light()
    + pn.theme(legend_key=pn.element_rect(color="white"))
    + pn.labs(x="$\\theta$", y="Mean absolute deviation", color=" ")
    + pn.geom_line(
        pn.aes(x="index", y="value", linetype="variable", color="variable"), size=1
    )
    + pn.scales.scale_linetype_manual(["solid", "--"], guide=False)
    + pn.scales.scale_color_manual(values=["#F8766D", "#fecd1a"],)
    + pn.theme(text=pn.element_text(family="times", size=18), legend_position="none")
)

plot_folder = get_folder("submit")
plot_MSE.save(
    f"{plot_folder}/intro_exponential_MSE.png",
    width=7,
    height=4.8,
    verbose=False,
    dpi=300,
)
plot_folder = get_folder("submit")
plot_MAD.save(
    f"{plot_folder}/intro_exponential_MAD.png",
    width=7,
    height=4.8,
    verbose=False,
    dpi=300,
)
