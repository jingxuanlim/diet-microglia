# Vendored from analysis_toolbox (visual.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def close_plot(fig):
    """Do everything you can to cleanly close a plot"""

    import gc

    axes = fig.axes
    for ax in axes:
        ax.clear()
    fig.clear()

    # Close the figure window
    plt.close(fig)

    gc.collect()


def get_transparent_cm(
    cmap,
    tvmax=0.8,
    diverging=False,
    gradient=True,
    func=lambda x: x,
    direction="ascending",
):
    """
    cmap [plt.cm.cmap or str]
    diverging [bool]: whether or not colormap diverges
    gradient [bool]: if not gradient, only the smallest value is transparent
    direction [str]: 'ascending' or 'descending'
                     if 'ascending', bigger numbers are less transparent
                     if 'descending', bigger numbers are more transparent
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import copy

    if type(cmap) == str:
        cmap = getattr(plt.cm, cmap)

    # copy colormap so that the original one remains unedited
    cmap = copy.deepcopy(cmap)

    cmap._init()

    # The last three rows are the colours for (1) data low and high,
    # (2) out-of-range values and (3) for masked values.

    if diverging:

        if gradient:
            alphas = func(np.linspace(0, tvmax, int(cmap.N / 2)))
            alphas = np.concatenate((np.flip(alphas), alphas))
            cmap._lut[:-3, -1] = alphas

        else:
            cmap._lut[int(cmap.N / 2), -1] = tvmax

    else:

        if gradient:
            alphas = np.linspace(0, tvmax, cmap.N)
            if direction == "ascending":
                cmap._lut[:-3, -1] = func(alphas)
            elif direction == "descending":
                cmap._lut[:-3, -1] = func(alphas[::-1])
        else:
            if direction == "ascending":
                cmap._lut[0, -1] = tvmax
            elif direction == "descending":
                cmap._lut[0, 0] = tvmax

    return cmap


def draw_contour(mask):

    import cv2 as cv

    # plot the ROIs as contours
    mask = mask.astype("uint8")
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    canvas = np.zeros_like(mask)
    cv.drawContours(canvas, contours, -1, (255, 255, 255), 1)

    return canvas


def blend_transparency(image_stack, alpha, cmap, vmin=None, vmax=None):
    """Blend transparency with colors
    Guide: https://matplotlib.org/gallery/images_contours_and_fields/image_transparency_blend.html

    """

    from matplotlib.colors import Normalize

    if vmin is None:
        vmin = np.nanmin(image_stack)
    if vmax is None:
        vmax = np.nanmax(image_stack)
    stack_norm = Normalize(
        vmin=vmin, vmax=vmax, clip=True
    )  # cluster labels normalized to 0-1
    colors = stack_norm(image_stack)

    # if there's no variation in alpha, everything is solid
    if np.nanmin(alpha) == np.nanmax(alpha):
        alpha_min = 0
        alpha_max = np.nanmax(alpha)
    else:
        alpha_min = np.nanmin(alpha)
        alpha_max = np.nanmax(alpha)

    alphas = Normalize(vmin=alpha_min, vmax=alpha_max, clip=True)(alpha)

    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(colors)
    colors[..., -1] = alphas

    return stack_norm, cmap, colors


def annotate_custom_pvalues(ax, data, x, y, order, box_pairs, pvalues, text_format='star', **kwargs):
    """
    Annotate an existing seaborn plot with custom p-values from external models.

    Parameters:
    - ax, data, x, y, order: Standard seaborn plotting arguments.
    - box_pairs: List of tuples representing the pairs to compare.
    - pvalues: List of p-values matching the box_pairs.
    - text_format: Format for the p-value annotations. Supported options:
        - "star" (default): significance brackets with asterisks.
        - "simple": simplified p-values.
        - "full": full p-values (e.g. p = 0.05).
    - **kwargs: Extra formatting parameters passed to statannotations.Annotator.configure
      (e.g., pvalue_format_string, simple_format_string, pvalue_thresholds, p_capitalized, p_separators).
    """
    from statannotations.Annotator import Annotator
    import numpy as np

    valid_formats = ['star', 'simple', 'full']
    if text_format not in valid_formats:
        raise ValueError(f"text_format must be one of {valid_formats}, got '{text_format}'")
    
    # --- NEW: Safely handle if 'ax' is returned as a NumPy array of axes ---
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()[0]
    
    # 1. Initialize the annotator on the existing axis
    annotator = Annotator(ax, box_pairs, data=data, x=x, y=y, order=order)
    
    # 2. Configure the aesthetics (test=None tells it NOT to calculate its own stats)
    annotator.configure(test=None, text_format=text_format, loc="inside", **kwargs)
    
    # 3. Inject your custom p-values
    annotator.set_pvalues(pvalues=pvalues)
    
    # 4. Draw the brackets and stars
    annotator.annotate()
    
    return ax
