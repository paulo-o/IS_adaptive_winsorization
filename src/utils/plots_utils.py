import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os


def _check_plotnine_grid(plots_list, figsize):
    if not type(plots_list) == list:
        raise ValueError("Input plots_list is not a list")
    if (not type(figsize) == tuple) or (not len(figsize) == 2):
        raise ValueError("Input figsize should be a tuple of length 2")


def plotnine_grid(
    plots_list,
    row=None,
    col=1,
    height=12,
    width=8,
    dpi=600,
    ratio=None,
    pixels=10000,
    figsize=(12, 8),
):
    """From George Sotiropoulos' answer in https://stackoverflow.com/questions/52331622/
    plotnine-any-work-around-to-have-two-plots-in-the-same-figure-and-print-it."""

    _check_plotnine_grid(plots_list, figsize)  # Check the input

    # Assign values that have not been provided based on others.
    # In the end, height and width should be provided.
    if row is None:
        row = len(plots_list)

    if ratio is None:
        ratio = 1.5 * col / row

    if height is None and width is not None:
        height = ratio * width

    if height is not None and width is None:
        width = height / ratio

    if height is None and width is None:
        area = pixels / dpi
        width = np.sqrt(area / ratio)
        height = ratio * width

    # Do actual subplot creation and plot output.
    i = 1
    fig = plt.figure(figsize=figsize)
    plt.autoscale(tight=True)
    for image_sel in plots_list:  # image_sel = plots_list[i]
        image_sel.save(
            "image" + str(i) + ".png",
            height=height,
            width=width,
            dpi=500,
            verbose=False,
        )
        fig.add_subplot(row, col, i)
        plt.imshow(img.imread("image" + str(i) + ".png"), aspect="auto")
        fig.tight_layout()
        fig.get_axes()[i - 1].axis("off")
        i = i + 1
        os.unlink(
            "image" + str(i - 1) + ".png"
        )  # os.unlink is basically os.remove but in some cases quicker
        fig.patch.set_visible(False)
        plt.axis('off')
    return fig
