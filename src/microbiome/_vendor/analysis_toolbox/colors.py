# Vendored from analysis_toolbox (colors.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np


def rescaleRGB(colorlist):
    """inputs a list of RBG values and returns a list of RGB values rescaled to
    matplotlib's specification"""
    return [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in colorlist]


tableau_trafficlight9_RGB = [
    (177, 3, 24),
    (219, 161, 58),
    (48, 147, 67),
    (216, 37, 36),
    (255, 193, 86),
    (105, 183, 100),
    (242, 108, 100),
    (255, 221, 113),
    (159, 205, 153),
]


tableau_trafficlight9 = rescaleRGB(tableau_trafficlight9_RGB)


def generate_random_color_list(num_colors, eight_byte=False):

    size = num_colors * 3
    rgb_elements = np.random.random(size=size)
    if eight_byte:
        rgb_elements = rgb_elements * 256
    return list(np.reshape(rgb_elements, (num_colors, 3)))


def generate_random_colormap(
    num_colors, fixed={}, cmap_name="", cmap_type="qualitative"
):
    """
    Generate a (mostly) random colormap
    """

    color_list = generate_random_color_list(num_colors)

    for index, color in fixed.items():
        color_list[index] = color

    return create_colormap_from_list(color_list, cmap_name, cmap_type=cmap_type)


def create_colormap_from_list(color_list, cmap_name, cmap_type="qualitative"):

    if cmap_type == "qualitative":

        from matplotlib.colors import ListedColormap

        return ListedColormap(color_list, name=cmap_name)

    elif cmap_type == "continuous":

        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(cmap_name, color_list)
