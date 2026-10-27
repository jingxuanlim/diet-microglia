# Vendored from analysis_toolbox (stacks_helper.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm_notebook as tqdm


def create_projections_fig(
    stack_shape=[52, 1144, 2048],
    voxel_spacing=[5, 0.406, 0.406],
    fig_scale=1,
    fig_width=None,
    dpi="auto",
    compact=False,
    single_var=False,
):

    stack_dims = np.multiply(np.array(stack_shape), np.array(voxel_spacing))

    # Create figure with layout
    import matplotlib.gridspec as gridspec

    rows = 4
    cols = 2

    wh_ratio = (
        (stack_dims[1] + stack_dims[0]),
        (stack_dims[2] + stack_dims[0] + (stack_dims[0] / 4)),
    )

    if fig_width is None:
        width = wh_ratio[0] / 100
        height = wh_ratio[1] / 100
    else:
        width = fig_width
        height = (wh_ratio[1] / wh_ratio[0]) * fig_width

    scaled_width = width * fig_scale
    scaled_height = height * fig_scale
    figsize = (scaled_width, scaled_height)

    if dpi == "auto":
        ## each pixel in the volume should be represented by
        ## at least a dot in the figure
        ## assuming the image only fills half of the width and
        ## we want 2x the amount of dots to pixels
        expected_width_ndot = (stack_shape[1] + stack_shape[0]) * 2 * 2
        dpi = int(np.ceil(expected_width_ndot / scaled_width))

    print(f"Creating figure of size {figsize} and dpi {dpi}")

    proj_fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)

    z = stack_dims[0]
    y = stack_dims[1]
    x = stack_dims[2]

    gs = gridspec.GridSpec(
        rows, cols, height_ratios=[x, z, x / 100, x / 10], width_ratios=[y, z]
    )
    xy_ax = plt.subplot(gs[:1, :1])
    xz_ax = plt.subplot(gs[:1, 1:])
    yz_ax = plt.subplot(gs[1:2, :1])

    if single_var:
        print("Creating subplot for variable...")
        var_ax = plt.subplot(gs[1:2, 1:], polar=True)
    else:
        print("Variable not demanded...")
        var_ax = None

    cb_ax = plt.subplot(gs[3:, :])

    if compact:
        if isinstance(compact, bool):
            spacing = 0
        else:
            spacing = compact
        gs.update(wspace=spacing, hspace=spacing)

    return proj_fig, gs, xy_ax, xz_ax, yz_ax, var_ax, cb_ax


def return_stack_contour(
    image_stack,
    debug=False,
    nbins_over_max=6,
    percentile=None,
    mask_mode="biggest plane",
):

    ## see: https://stackoverflow.com/a/50128836
    if not image_stack.flags["C_CONTIGUOUS"]:
        image_stack = np.ascontiguousarray(image_stack)

    fig, ax = plt.subplots()
    bg_n, bg_bins, bg_patches = plt.hist(np.ravel(image_stack), bins=200)

    if nbins_over_max is not None:
        bg_noise = bg_bins[np.argmax(bg_n) + nbins_over_max]
    if percentile is not None:
        bg_noise = np.nanpercentile(image_stack, percentile)

    ax.axvline(x=bg_noise, linestyle="--", color="k")

    from microbiome._vendor.analysis_toolbox.visual import draw_contour

    bg_mask = image_stack > bg_noise

    if mask_mode == "biggest plane":
        biggest_mask = np.argmax(bg_mask.sum(axis=(1, 2)))
        contours = [draw_contour(bg_mask[i]) for i in range(bg_mask.shape[0])]

        if debug:
            from microbiome._vendor.analysis_toolbox.visual import get_transparent_cm

            contour_cmap = get_transparent_cm("YlGn", tvmax=1, gradient=False)

            plt.figure()
            plt.imshow(contours[biggest_mask], cmap=contour_cmap)

    elif mask_mode == "maximum projection":

        bg_xy_max = np.nanmax(image_stack, axis=0)
        bg_xz_max = np.nanmax(image_stack, axis=1)
        bg_yz_max = np.nanmax(image_stack, axis=2)

        bg_xy_mask = bg_xy_max > bg_noise
        bg_xz_mask = bg_xz_max > bg_noise
        bg_yz_mask = bg_yz_max > bg_noise

        bg_xy_contour = draw_contour(bg_xy_mask)
        bg_xz_contour = draw_contour(bg_xz_mask)
        bg_yz_contour = draw_contour(bg_yz_mask)

        contours = [bg_xy_contour, bg_xz_contour, bg_yz_contour]

        if debug:
            from microbiome._vendor.analysis_toolbox.stacks_helper import create_projections_fig
            from microbiome._vendor.analysis_toolbox.visual import get_transparent_cm

            trans_contour = get_transparent_cm("binary", gradient=False)
            fig, gs, xy_ax, xz_ax, yz_ax, var_ax, cb_ax = create_projections_fig(
                image_stack.shape, dpi=60
            )
            xy_ax.imshow(bg_xy_contour.T, aspect="auto", cmap=trans_contour)
            xz_ax.imshow(bg_xz_contour.T, aspect="auto", cmap=trans_contour)
            yz_ax.imshow(bg_yz_contour, aspect="auto", cmap=trans_contour)

        return contours


def make_projections(
    image_stack,
    voxel_spacing=(5, 0.40625, 0.40625),
    alpha=1,
    amin_mode="raw",
    amin=None,
    amax_mode="raw",
    amax=None,
    max_p=True,
    min_p=False,
    perspective=False,
    cmap="RdBu_r",
    style="default",
    vmin_mode="raw",
    vmin=None,
    vmax_mode="raw",
    vmax=None,
    oom_diff_thresh=4,
    fig_scale=1,
    fig_width=2,
    dpi="auto",
    cmap_contour="binary_r",
    contour=None,
    background=None,
    background_vmin=30,
    background_vmax=100,
    background_vmode="percentile",
    background_clahe=False,
    rotate180=False,
    flip_over=False,
    proj_fig=None,
    gs=None,
    xy_ax=None,
    xz_ax=None,
    yz_ax=None,
    cb_ax=None,
    compact=False,
    xlabel="",
    suptitle="",
    savename="",
    closeplot=False,
    debug=False,
):
    """
    Make a brain stack projection. Currently, only perspective mode is
    supported when all the alphas are given.

    TODO: Include aspect ratio as part of the plot?

    Parameters
    ----------
    image_stack [np.array]: brain stack to be plotted
    alpha [int/np.array]: use one alpha value for all or one for every
                          pixel
    TODO: the maximum/mean/minumum alpha is not the right for the
          maximum/mean/minumum image_stack!!!! (also fix make_slabs)
    max_p [bool]: maximum projection?
    min_p [bool]: minimum projection?
    perspective [bool]: plane by plane plotting; what you'd see if you
                        were to look from the top, side, or back
    cmap [str]: colormap
    vmin [numeric]: imshow vmin
    vmax [numeric]: imshow vmax
    fig_scale [numeric]:
    dpi [int]:
    cmap_contour [str]:
    contour:
    background [np.array]: stack of the background
    xlabel [str]:
    suptitle [str]:
    savename [str]:

    Returns
    -------
    """

    # calculate extent
    z_extent, y_extent, x_extent = np.array(image_stack.shape) * np.array(voxel_spacing)

    plt.style.use(style)

    finite_image_stack = image_stack[np.isfinite(image_stack)]

    # if image stack only has ones, then it must be binary mask
    image_stack_vals = np.unique(finite_image_stack)
    if (
        len(image_stack_vals) == 1
        and image_stack_vals[0] == 1
        and vmin is None
        and vmax is None
    ):
        vmin = 0

    # image_stack = np.nan_to_num(image_stack, nan=np.nan, posinf=np.nan, neginf=np.nan)
    if vmin_mode == "percentile":

        if vmin is not None:
            vmin = np.nanpercentile(image_stack, vmin)

    if vmax_mode == "percentile":

        if vmax is not None:
            vmax = np.nanpercentile(image_stack, vmax)

    if debug:
        print(f"vmin: {vmin}  vmax: {vmax}")

    # determine whether or not to lognorm by assesssing order of magnitude difference
    from microbiome._vendor.analysis_toolbox.math_helper import (
        percentile_order_of_magnitude_difference,
        order_of_magnitude_difference,
    )

    if vmin is not None and vmax is not None:
        oom_diff = order_of_magnitude_difference(vmax, vmin)
    else:
        oom_diff = percentile_order_of_magnitude_difference(finite_image_stack, 100, 0)

    if oom_diff > oom_diff_thresh:

        if debug:
            print("Attempt to use SymLogNorm()")
            print(f"Right before calling SymLogNorm: {vmin}, {vmax}")

        norm = mpl.colors.SymLogNorm(linthresh=1e-2, vmin=vmin, vmax=vmax)

        from microbiome._vendor.analysis_toolbox.math_helper import check_same_sign

        # if np.any(finite_image_stack == 0):
        # # if check_same_sign(finite_image_stack):

        # #     print("Using LogNorm()")

        # #     if np.all(finite_image_stack < 0):

        # #         import warnings
        # #         warnings.warn("Converting image_stack to positive as log of anything 0 and below is nan")
        # #         print(f"Old vmin vmax: {vmin} {vmax}")

        # #         image_stack = -image_stack
        # #         vmin = np.abs(vmin)
        # #         vmax = np.abs(vmax)

        # #         # if '_r' in cmap: cmap.replace('_r', '')
        # #         # else: cmap =+ '_r'

        #         # print(f"New vmin vmax: {vmin} {vmax}")

        #     norm = mpl.colors.SymLogNorm(vmin=vmin, vmax=vmax)

        # else:

        #     print("Finite values are not the same sign or has zeros...")
        #     print("using Normalize()")
        #     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    else:
        if debug:
            print("using Normalize()")
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if rotate180:
        image_stack = np.flip(image_stack, axis=2)
        if background is not None:
            background = np.flip(background, axis=2)
        if type(alpha).__module__ == np.__name__:
            alpha = np.flip(alpha, axis=2)

    if flip_over:
        image_stack = np.flip(image_stack, axis=0)
        if background is not None:
            background = np.flip(background, axis=0)
        if type(alpha).__module__ == np.__name__:
            alpha = np.flip(alpha, axis=0)

    from scipy.ndimage import rotate

    if [x for x in (proj_fig, gs, xy_ax, xz_ax, yz_ax, cb_ax) if x is None]:
        stack_dimensions = np.multiply(np.array(image_stack.shape), voxel_spacing)
        if debug:
            print(f"stack_dimensions: {stack_dimensions}")

        print("Creating new figure...")
        proj_fig, gs, xy_ax, xz_ax, yz_ax, var_ax, cb_ax = create_projections_fig(
            stack_shape=image_stack.shape,
            voxel_spacing=voxel_spacing,
            fig_scale=fig_scale,
            fig_width=fig_width,
            dpi=dpi,
            compact=compact,
        )

    if perspective:

        # alpha_xy, alpha_xz, alpha_yz not used
        proj_types = ["perspective"]

    # if both max and min projections (currently doesn't work as don't have alpha)
    elif np.logical_and(max_p, min_p):

        proj_types = ["max", "min"]

    elif np.logical_or(max_p, min_p):  # if max or mean

        if min_p:
            proj_types = ["min"]
        elif max_p:
            proj_types = ["max"]

    else:  # neither maxp or minp and not layered

        proj_types = ["mean"]

    if contour is not None:

        bg_xy_contour, bg_xz_contour, bg_yz_contour = return_stack_contour(
            contour, nbins_over_max=None, percentile=80, mask_mode="maximum projection"
        )
        from microbiome._vendor.analysis_toolbox.visual import get_transparent_cm

        trans_cmap_contour = get_transparent_cm(cmap_contour, gradient=False)

        xy_ax.imshow(
            bg_xy_contour.T,
            cmap=trans_cmap_contour,
            alpha=1,
            interpolation="none",
            extent=[0, y_extent, x_extent, 0],
            aspect="equal",
        )
        xz_ax.imshow(
            bg_xz_contour.T,
            cmap=trans_cmap_contour,
            alpha=1,
            interpolation="none",
            extent=[0, z_extent, x_extent, 0],
            aspect="equal",
        )
        yz_ax.imshow(
            np.flipud(bg_yz_contour),
            cmap=trans_cmap_contour,
            alpha=1,
            interpolation="none",
            extent=[0, y_extent, z_extent, 0],
            aspect="equal",
        )

    if background is not None:

        bg_xy = np.nanmax(background, axis=0)
        bg_xz = np.nanmax(background, axis=1)
        bg_yz = np.nanmax(background, axis=2)

        if background_clahe:

            from microbiome._vendor.analysis_toolbox.image_helper import apply_clahe

            bg_xy = apply_clahe(bg_xy)
            bg_xz = apply_clahe(bg_xz)
            bg_yz = apply_clahe(bg_yz)

        if background_vmode == "percentile":
            if background_vmin is not None:
                background_vmin = np.nanpercentile(background, background_vmin)
            if background_vmax is not None:
                background_vmax = np.nanpercentile(background, background_vmax)

        xy_ax.imshow(
            bg_xy.T,
            cmap="gray",
            vmin=background_vmin,
            vmax=background_vmax,
            alpha=1,
            interpolation="none",
            extent=[0, y_extent, x_extent, 0],
            aspect="equal",
        )
        xz_ax.imshow(
            bg_xz.T,
            cmap="gray",
            vmin=background_vmin,
            vmax=background_vmax,
            alpha=1,
            interpolation="none",
            extent=[0, z_extent, x_extent, 0],
            aspect="equal",
        )
        # rotate only works when there are no nans because it relies on interpolation
        yz_ax.imshow(
            np.fliplr(rotate(bg_yz, 180)),
            cmap="gray",
            vmin=background_vmin,
            vmax=background_vmax,
            alpha=1,
            interpolation="none",
            extent=[0, y_extent, z_extent, 0],
            aspect="equal",
        )

    if perspective:

        # x and y are of much higher value than z, reduce the alpha
        # when plotting from the side and front so that they have
        # comparable alphas from the top
        y_alphagain = image_stack.shape[0] / image_stack.shape[1]
        x_alphagain = image_stack.shape[0] / image_stack.shape[2]

        # if the alpha for every pixel is provided (in an numpy array),
        # we need to use RGBA values to plot
        if type(alpha).__module__ == np.__name__:

            from microbiome._vendor.analysis_toolbox.visual import blend_transparency

            stack_norm, cmap, colors = blend_transparency(
                image_stack, alpha, cmap, vmin=vmin, vmax=vmax
            )

            y_tempstack = colors.copy()
            y_tempstack[..., -1] = y_tempstack[..., -1] * y_alphagain

            x_tempstack = colors.copy()
            x_tempstack[..., -1] = x_tempstack[..., -1] * x_alphagain

            for zplane in tqdm(range(image_stack.shape[0])):

                xy_imshow = xy_ax.imshow(
                    colors[zplane, :, :, :].transpose(1, 0, 2),
                    interpolation="none",
                    extent=[0, y_extent, x_extent, 0],
                    aspect="equal",
                )

            for yplane in tqdm(range(image_stack.shape[1])):

                xz_imshow = xz_ax.imshow(
                    y_tempstack[:, yplane, :, :].transpose(1, 0, 2),
                    interpolation="none",
                    extent=[0, z_extent, x_extent, 0],
                    aspect="equal",
                )

            for xplane in tqdm(range(image_stack.shape[2])):

                yz_imshow = yz_ax.imshow(
                    np.flipud(x_tempstack[:, :, xplane, :]),
                    interpolation="none",
                    extent=[0, y_extent, z_extent, 0],
                    aspect="equal",
                )

        else:

            for zplane in tqdm(range(image_stack.shape[0])):

                xy_imshow = xy_ax.imshow(
                    image_stack[zplane, :, :].T,
                    cmap=cmap,
                    norm=norm,
                    alpha=alpha,
                    interpolation="none",
                    extent=[0, y_extent, x_extent, 0],
                    aspect="equal",
                )

            for yplane in tqdm(range(image_stack.shape[1])):

                xz_imshow = xz_ax.imshow(
                    image_stack[:, yplane, :].T,
                    cmap=cmap,
                    norm=norm,
                    alpha=alpha * y_alphagain,
                    interpolation="none",
                    extent=[0, z_extent, x_extent, 0],
                    aspect="equal",
                )

            for xplane in tqdm(range(image_stack.shape[2])):

                yz_imshow = yz_ax.imshow(
                    np.flipud(image_stack[:, :, xplane]),
                    cmap=cmap,
                    norm=norm,
                    alpha=alpha * x_alphagain,
                    interpolation="none",
                    extent=[0, y_extent, z_extent, 0],
                    aspect="equal",
                )

    else:

        from microbiome._vendor.analysis_toolbox.utils import apply_to_each_and_every

        for proj_type in proj_types:

            if type(alpha).__module__ == np.__name__:

                if amin_mode == "percentile":

                    if amin is not None:
                        amin = np.nanpercentile(alpha, amin)

                if amax_mode == "percentile":

                    if amax is not None:
                        amax = np.nanpercentile(alpha, amax)

                from microbiome._vendor.analysis_toolbox.utils import (
                    index_based_on_projection,
                    zscore_and_multiply,
                )

                ## index based on norm_weighted_image_stack
                norm_weighted_image_stack = zscore_and_multiply(
                    [alpha]
                )  # [image_stack, alpha]

                from microbiome._vendor.analysis_toolbox.utils import normalise

                norm_alpha = normalise(
                    np.clip(alpha, a_min=amin, a_max=amax), a_min=amin, a_max=amax
                )
                print(np.nanmax(norm_alpha))
                print(np.nanmin(norm_alpha))

                xy_proj, xy_alpha = index_based_on_projection(
                    norm_weighted_image_stack,
                    axis=0,
                    arrs_to_index=[image_stack, norm_alpha],
                    proj_type=proj_type,
                )
                # alpha must be between 0 and 1
                xy_alpha = np.nan_to_num(xy_alpha)

                xz_proj, xz_alpha = index_based_on_projection(
                    norm_weighted_image_stack,
                    axis=1,
                    arrs_to_index=[image_stack, norm_alpha],
                    proj_type=proj_type,
                )
                xz_alpha = np.nan_to_num(xz_alpha)

                yz_proj, yz_alpha = index_based_on_projection(
                    norm_weighted_image_stack,
                    axis=2,
                    arrs_to_index=[image_stack, norm_alpha],
                    proj_type=proj_type,
                )
                # alpha must be between 0 and 1
                yz_alpha = np.nan_to_num(yz_alpha)

                xy_alpha = xy_alpha.T
                xz_alpha = xz_alpha.T
                yz_alpha = np.flipud(yz_alpha)  # alpha must be between 0 and 1

            else:
                xy_proj = getattr(np, f"nan{proj_type}")(image_stack, axis=0)
                xz_proj = getattr(np, f"nan{proj_type}")(image_stack, axis=1)
                yz_proj = getattr(np, f"nan{proj_type}")(image_stack, axis=2)

                xy_alpha = alpha
                xz_alpha = alpha
                yz_alpha = alpha

            if vmin is None:
                vmin = apply_to_each_and_every([xy_proj, xz_proj, yz_proj], np.nanmin)
            if vmax is None:
                vmax = apply_to_each_and_every([xy_proj, xz_proj, yz_proj], np.nanmax)

            # to save as svg, interpolation='nearest' is necessary
            # https://stackoverflow.com/questions/69914117/matplotlib-issue-when-saving-imshow-as-pdf-with-interpolation-and-alpha
            xy_imshow = xy_ax.imshow(
                xy_proj.T,
                cmap=cmap,
                norm=norm,
                alpha=xy_alpha,
                interpolation="nearest",
                extent=[0, y_extent, x_extent, 0],
                aspect="equal",
            )
            xz_imshow = xz_ax.imshow(
                xz_proj.T,
                cmap=cmap,
                norm=norm,
                alpha=xz_alpha,
                interpolation="nearest",
                extent=[0, z_extent, x_extent, 0],
                aspect="equal",
            )
            yz_imshow = yz_ax.imshow(
                np.flipud(yz_proj),
                cmap=cmap,
                norm=norm,
                alpha=yz_alpha,
                interpolation="nearest",
                extent=[0, y_extent, z_extent, 0],
                aspect="equal",
            )

    proj_type_str = "+".join(proj_types)

    if not compact:
        xy_ax.set_title(proj_type_str + " xy projection")
        xz_ax.set_title(proj_type_str + " xz projection")
        yz_ax.set_title(proj_type_str + " yz projection")

    else:
        suptitle += f" ({proj_type_str})"

    if type(alpha).__module__ == np.__name__ and perspective:
        colorbar = proj_fig.colorbar(
            mpl.cm.ScalarMappable(norm=stack_norm, cmap=cmap),
            cax=cb_ax,
            orientation="horizontal",
        )

    else:
        # proj_fig.colorbar(yz_imshow,cax=cb_ax,orientation='horizontal')
        # proj_fig.colorbar(xz_imshow,cax=cb_ax,orientation='horizontal')
        colorbar = proj_fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cb_ax,
            orientation="horizontal",
        )
        # one should enough as they all have the same vmin and vmax

    xy_ax.set_aspect("equal")
    xz_ax.set_aspect("equal")
    yz_ax.set_aspect("equal")

    # xy_ax.set_aspect(voxel_spacing[1]/voxel_spacing[2])
    # xz_ax.set_aspect(voxel_spacing[2]/voxel_spacing[0])
    # yz_ax.set_aspect(voxel_spacing[0]/voxel_spacing[1])

    xy_ax.axis("off")
    xz_ax.axis("off")
    yz_ax.axis("off")

    cb_ax.set_xlabel(xlabel)

    # if compact: colorbar.set_ticks([])

    proj_fig.suptitle(suptitle)
    # proj_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if savename:
        print(savename)

        if isinstance(savename, str):
            proj_fig.savefig(savename, bbox_inches='tight')
        elif isinstance(savename, list):
            for sn in savename:
                proj_fig.savefig(sn, bbox_inches='tight')

    # plt.style.use('default')

    if closeplot:
        plt.close(proj_fig)
    else:
        proj_ax = np.array([xy_ax, xz_ax, yz_ax, cb_ax])

        if type(alpha).__module__ == np.__name__ and perspective:
            return colors, cmap, proj_fig, proj_ax, gs
        else:
            return proj_fig, proj_ax, gs


def divide_into_slabs(image_stack, num_slabs, axis=0, return_slabinfo=False):
    """

    Parameters
    ----------
    image_stack : stack with dimensions [z, y, x]
    num_slabs : number of slabs, or slab starts

    Returns
    -------
    slabs : list of array representing slabs
    slab_planes: number of planes per slab
    slab_starts: index of the first plane in each slab

    """

    if isinstance(num_slabs, list) or isinstance(num_slabs, np.ndarray):
        slab_starts = num_slabs
    else:
        slab_planes = int(np.ceil(image_stack.shape[axis] / num_slabs))
        slab_starts = np.arange(0, image_stack.shape[axis], slab_planes)

    slabs = []

    for iplane, plane in enumerate(slab_starts):

        start = slab_starts[iplane]
        try:
            end = slab_starts[iplane + 1]
        except:
            end = image_stack.shape[axis]

        slice_range = np.arange(start, end)
        slabs.append(np.take(image_stack, slice_range, axis=axis))

    if isinstance(num_slabs, list) or isinstance(num_slabs, np.ndarray):
        slab_planes = [len(planes) for planes in slabs]

    if return_slabinfo:
        return slabs, slab_planes, slab_starts
    else:
        return slabs


def make_slabs(
    image_stack,
    num_slabs,
    alpha=1,
    max_p=True,
    min_p=False,
    perspective=False,
    cmap="RdBu_r",
    style="default",
    compact=False,
    vmin_mode="raw",
    vmin=None,
    vmax_mode="raw",
    vmax=None,
    oom_diff_thresh=1000,
    fig_scale=0.8,
    dpi=100,
    cmap_contour="viridis",
    contour=[],
    background=None,
    background_vmin=None,
    background_vmax=None,
    background_vmode="percentile",
    background_clahe=False,
    rotate180=False,
    flip_over=False,
    xlabel="",
    suptitle="",
    slab_titles=[],
    savename="",
    debug=False,
):
    """
    Make a brain stack projection. Currently, only perspective mode is
    supported when all the alphas are given.

    Parameters
    ----------
    image_stack [np.array]: brain stack to be plotted
    alpha [int/np.array]: use one alpha value for all or one for every
                          pixel
    max_p [bool]: maximum projection?
    min_p [bool]: minimum projection?
    perspective [bool]: plane by plane plotting; what you'd see if you
                        were to look from the top, side, or back
    cmap [str]: colormap
    vmin [numeric]: imshow vmin
    vmax [numeric]: imshow vmax
    fig_scale [numeric]:
    dpi [int]:
    cmap_contour [str]:
    contour:
    background [np.array]: stack of the background
    xlabel [str]:
    suptitle [str]:
    slab_titles [list]: list of strings, save length with `num_slabs`
    savename [str]:

    Returns
    -------
    """
    plt.style.use(style)

    finite_image_stack = image_stack[np.isfinite(image_stack)]
    # if image stack only has ones, then it must be binary mask
    image_stack_vals = np.unique(finite_image_stack)
    if (
        len(image_stack_vals) == 1
        and image_stack_vals[0] == 1
        and vmin is None
        and vmax is None
    ):
        vmin = 0

    if vmin_mode == "percentile":
        if vmin is not None:
            vmin = np.nanpercentile(image_stack, vmin)

    if vmax_mode == "percentile":
        if vmax is not None:
            vmax = np.nanpercentile(image_stack, vmax)

    # determine whether or not to lognorm by assesssing order of magnitude difference
    from microbiome._vendor.analysis_toolbox.math_helper import order_of_magnitude

    oom_diff = order_of_magnitude(
        np.abs(
            np.nanpercentile(finite_image_stack, 98)
            - np.nanpercentile(finite_image_stack, 2)
        )
    )
    if oom_diff > oom_diff_thresh:

        if debug:
            print("Using LogNorm()")
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        if debug:
            print("Using Normalize()")
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if rotate180:
        image_stack = np.flip(image_stack, axis=2)
        if background is not None:
            background = np.flip(background, axis=2)
        if type(alpha).__module__ == np.__name__:
            alpha = np.flip(alpha, axis=2)

    if flip_over:
        image_stack = np.flip(image_stack, axis=0)
        if backgorund is not None:
            background = np.flip(background, axis=0)
        if type(alpha).__module__ == np.__name__:
            alpha = np.flip(alpha, axis=0)
        slab_titles = list(np.flip(slab_titles))

    from scipy.ndimage import rotate

    # Create figure with layout
    import matplotlib.gridspec as gridspec

    rows = 8
    if isinstance(num_slabs, list) or isinstance(num_slabs, np.ndarray):
        cols = 4 * len(num_slabs)
    else:  # is numeric
        cols = 4 * num_slabs

    proj_fig, proj_ax = plt.subplots(
        1, int(cols / 4), figsize=(cols * fig_scale, rows * fig_scale), dpi=dpi
    )
    if compact:
        plt.subplots_adjust(wspace=0, hspace=0)

    if perspective:

        proj_types = "perspective"

    # if both max and min projections (currently doesn't work as don't have alpha)
    elif np.logical_and(max_p, min_p):

        proj_types = ["max", "min"]

    elif np.logical_or(max_p, min_p):  # if max or mean

        if min_p:
            proj_types = ["min"]
        elif max_p:
            proj_types = ["max"]

    else:  # neither maxp or minp and not layered

        proj_types = ["mean"]

    # divide the image stack and background stack up into the
    # indicated number of planes
    image_slabs, slab_planes, slab_starts = divide_into_slabs(
        image_stack, num_slabs, return_slabinfo=True
    )

    if background is not None:
        background_slabs = divide_into_slabs(background, num_slabs)
    else:
        background_slabs = [np.array([])] * len(slab_starts)

    if type(alpha).__module__ == np.__name__:
        alpha_slabs = divide_into_slabs(alpha, num_slabs)
    else:
        alpha_slabs = [np.array([])] * len(slab_starts)

    # for each slabs
    for i, (im, bg, ap, ax) in enumerate(
        zip(image_slabs, background_slabs, alpha_slabs, proj_ax)
    ):

        # Plot background
        if background is not None:

            bg_max = np.nanmax(bg, axis=0)
            if background_clahe:
                from microbiome._vendor.analysis_toolbox.image_helper import apply_clahe

                bg_max = apply_clahe(bg_max)

            if background_vmode == "percentile":
                if background_vmin is not None:
                    background_vmin = np.nanpercentile(background, background_vmin)
                if background_vmax is not None:
                    background_vmax = np.nanpercentile(background, background_vmax)

            ax.imshow(
                bg_max.T,
                aspect="auto",
                cmap="gray",
                vmin=background_vmin,
                vmax=background_vmax,
                alpha=1,
                interpolation="none",
            )

        # Plot image
        if perspective:

            # if the alpha for every pixel is provided (in an numpy array),
            # we need to use RGBA values to plot
            if type(alpha).__module__ == np.__name__:

                from microbiome._vendor.analysis_toolbox.visual import blend_transparency

                stack_norm, cmap, colors = blend_transparency(
                    im, ap, cmap, vmin=vmin, vmax=vmax
                )

                for zplane in tqdm(range(im.shape[0])):

                    xy_imshow = ax.imshow(
                        colors[zplane, :, :, :].transpose(1, 0, 2),
                        aspect="auto",
                        interpolation="none",
                    )

            else:

                for zplane in tqdm(range(im.shape[0])):

                    xy_imshow = ax.imshow(
                        im[zplane, :, :].T,
                        aspect="auto",
                        interpolation="none",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )

        else:

            for proj_type in proj_types:

                if type(alpha).__module__ == np.__name__:

                    from microbiome._vendor.analysis_toolbox.utils import (
                        zscore_and_multiply,
                        index_based_on_projection,
                        normalise,
                    )

                    norm_weighted_image_stack = zscore_and_multiply([im, ap])

                    norm_ap = normalise(ap)

                    xy_proj, xy_alpha = index_based_on_projection(
                        norm_weighted_image_stack,
                        axis=0,
                        arrs_to_index=[im, norm_ap],
                        proj_type=proj_type,
                    )
                    # alpha must be between 0 and 1
                    xy_alpha = np.nan_to_num(xy_alpha)

                else:

                    xy_proj = getattr(np, f"nan{proj_type}")(im, axis=0)
                    xy_alpha = ap

                if vmin is None:
                    vmin = np.nanmin(xy_proj)
                if vmax is None:
                    vmin = np.nanmax(xy_proj)

                xy_imshow = ax.imshow(
                    xy_proj.T,
                    aspect="auto",
                    interpolation="none",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=xy_alpha,
                )

        if len(contour) != 0:

            bg_xy_contour = contour[0]
            ax.imshow(
                bg_xy_contour.T,
                aspect="auto",
                cmap=cmap_contour,
                alpha=alpha,
                interpolation="none",
            )

        start = slab_starts[i]
        try:
            end = slab_starts[i + 1] - 1
        except:
            end = image_stack.shape[0] - 1

        proj_type_str = "+".join(proj_types)
        if not compact:
            if slab_titles:
                ax.set_title(
                    proj_type_str
                    + f" planes {start}-{end} of {image_stack.shape[0]-1}\n{slab_titles[start]}-{slab_titles[end]}"
                )
            else:
                ax.set_title(
                    proj_type_str + f" planes {start}-{end} of {image_stack.shape[0]-1}"
                )

        ax.axis("off")

        if type(alpha).__module__ == np.__name__ and perspective:
            colorbar = proj_fig.colorbar(
                mpl.cm.ScalarMappable(norm=stack_norm, cmap=cmap),
                ax=ax,
                orientation="horizontal",
            )

        else:
            colorbar = proj_fig.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
                ),
                ax=ax,
                orientation="horizontal",
            )

        colorbar.set_label(xlabel)

        if compact:
            colorbar.set_ticks([])

    if not compact:
        proj_fig.tight_layout()

    if slab_titles:
        proj_fig.subplots_adjust(top=0.85)  # more space for multiline titles
    else:
        proj_fig.subplots_adjust(top=0.9)
    proj_fig.suptitle(suptitle)

    if savename:

        proj_fig.savefig(savename)

    plt.style.use("default")

    return image_slabs, background_slabs, alpha_slabs, proj_fig, proj_ax


def convert_to_list_of_coords(coords):

    if isinstance(coords, np.ndarray):

        return list(coords.T)

    elif isinstance(coords, pd.Series):

        ## Such as the output from ski.measure.reggionprops_table
        return list(np.concatenate(coords.to_list()).T)


def plot_planes(
    stack,
    cols=10,
    compact=False,
    cmap="viridis",
    colorbar=True,
    colorbar_label="",
    stack_min=None,
    stack_max=None,
    percentile_min=0,
    percentile_max=100,
    figsize_scale=1,
    yx_ratio=1,
    alpha=None,
    suptitle="",
    savename="",
    closeplot=False,
):

    rows = np.ceil(stack.shape[0] / cols).astype("int")
    print(f"Rows: {rows}\nColumns: {cols}")

    figsize = (
        cols * figsize_scale,
        rows * figsize_scale * (stack.shape[1] / stack.shape[2]),
    )
    print(f"figsize: {figsize}")

    if compact:
        fig, ax = plt.subplots(
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"wspace": -0.00000001, "hspace": 0},
            constrained_layout=True,
        )
    else:
        fig, ax = plt.subplots(
            rows, cols, figsize=figsize, squeeze=False, constrained_layout=True
        )

    if not compact:
        for x in np.ravel(ax):
            x.axis("off")

    if stack_min is None:
        stack_min = np.nanpercentile(stack, percentile_min)
    if stack_max is None:
        stack_max = np.nanpercentile(stack, percentile_max)

    if stack_min == stack_max:

        if stack_min == 0:

            stack_min = stack_min - 0.01
            stack_max = stack_max + 0.01

        else:

            stack_min = stack_min * 0.99
            stack_max = stack_max * 1.01

    print(stack_min, stack_max)

    if (stack.shape[2] / stack.shape[1]) == 1:
        aspect = "equal"
    else:
        aspect = "auto"

    # if the alpha for every pixel is provided (in an numpy array),
    # we need to use RGBA values to plot
    if type(alpha).__module__ == np.__name__:

        from microbiome._vendor.analysis_toolbox.visual import blend_transparency

        stack_norm, cmap, colors = blend_transparency(
            stack, alpha, cmap, vmin=stack_min, vmax=stack_max
        )

        for i, field in enumerate(colors):

            row = i // cols
            col = i % cols

            im = ax[row, col].imshow(field, aspect=aspect)

            if not compact:
                ax[row, col].set_title(i)
            else:
                xax = ax[row, col].axes.get_xaxis()
                xax = xax.set_visible(False)
                yax = ax[row, col].axes.get_yaxis()
                yax = yax.set_visible(False)

    else:

        for i, field in enumerate(stack):

            row = i // cols
            col = i % cols

            im = ax[row, col].imshow(
                field, vmin=stack_min, vmax=stack_max, cmap=cmap, aspect=aspect
            )

            if not compact:
                ax[row, col].set_title(i)
            else:
                xax = ax[row, col].axes.get_xaxis()
                xax = xax.set_visible(False)
                yax = ax[row, col].axes.get_yaxis()
                yax = yax.set_visible(False)

    # make the rest of the axes disappear
    if compact:

        for col_ in np.arange(col + 1, cols):

            ax[row, col_].axis("off")

        # fig.subplots_adjust(wspace=0, hspace=0)

    if colorbar:

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

        if type(alpha).__module__ == np.__name__:
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=stack_norm, cmap=cmap), cax=cbar_ax
            )

        else:
            cbar = fig.colorbar(im, cax=cbar_ax)

    if colorbar and colorbar_label:
        cbar.set_label(colorbar_label)
    if suptitle:
        fig.suptitle(suptitle)

    if colorbar:
        fig.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    else:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if savename:
        fig.savefig(savename)
    if closeplot:
        plt.close(fig)

    return fig, ax


def stack_comparison_long(
    stacks_list,
    titles=[],
    color_list="viridis",
    vmins=[],
    vmaxs=[],
    combine_mode="overlay",
    show_engine="imshow",
    figsize_scale=1,
    savename="",
    closeplot=False,
):

    if isinstance(color_list, str):
        color_list = [color_list] * len(stacks_list)
    import numbers

    if isinstance(vmins, numbers.Number):
        vmins = [vmins] * len(stacks_list)
    if isinstance(vmaxs, numbers.Number):
        vmaxs = [vmaxs] * len(stacks_list)

    if not vmins:
        vmins = [None] * len(stacks_list)
    if not vmaxs:
        vmaxs = [None] * len(stacks_list)

    first_stack = stacks_list[0]
    ncols = len(stacks_list) * 2 - 1
    nrows = first_stack.shape[0]

    width = ncols * first_stack.shape[2] / 1000
    height = nrows * first_stack.shape[1] / 1000

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(width * figsize_scale, height * figsize_scale)
    )

    for x in np.ravel(ax):
        x.axis("off")

    from microfilm.microplot import microshow

    # zip through all planes
    for stack_ind, stack in enumerate(stacks_list):

        if stack_ind != 0 and combine_mode != "overlay":

            np_func = getattr(np, combine_mode)
            combined_stack = np_func(stack, stacks_list[stack_ind - 1])

        for plane_ind, plane in enumerate(stack):

            stack_col = stack_ind * 2

            if show_engine == "imshow":
                ax[plane_ind, stack_col].imshow(
                    plane,
                    alpha=1,
                    vmin=vmins[stack_ind],
                    vmax=vmaxs[stack_ind],
                    cmap=color_list[stack_ind],
                )

            elif show_engine == "microshow":
                microshow(images=[plane], ax=ax[plane_ind, stack_col])

            if plane_ind == 0:
                ax[plane_ind, stack_col].set_title(titles[stack_ind])

            if stack_col != 0:

                if combine_mode == "overlay":

                    if show_engine == "imshow":
                        ax[plane_ind, stack_col - 1].imshow(
                            stacks_list[stack_ind - 1][plane_ind],
                            alpha=1,
                            vmin=vmins[stack_ind - 1],
                            vmax=vmaxs[stack_ind - 1],
                            cmap=color_list[stack_ind - 1],
                        )
                        ax[plane_ind, stack_col - 1].imshow(
                            plane,
                            alpha=0.5,
                            vmin=vmins[stack_ind],
                            vmax=vmaxs[stack_ind],
                            cmap=color_list[stack_ind],
                        )

                    elif show_engine == "microshow":

                        microshow(
                            images=[stacks_list[stack_ind - 1][plane_ind], plane],
                            ax=ax[plane_ind, stack_col - 1],
                        )

                else:

                    if show_engine == "imshow":

                        ax[plane_ind, stack_col - 1].imshow(
                            combined_stack[plane_ind], vmin=0, cmap="inferno"
                        )

                    elif show_engine == "microshow":

                        microshow(
                            images=[combined_stack[plane_ind]],
                            ax=ax[plane_ind, stack_col - 1],
                        )

    if savename:
        fig.savefig(savename)
    if closeplot:
        plt.close(fig)

    return fig, ax


def load_defined_stack(stack_path):
    """
    Load stacks that also contain metadata
    """

    path_components = stack_path.split(".")

    extension_components = path_components[-(len(path_components) - 1) :]

    extension = ".".join(extension_components)

    if extension == "nii.gz":

        import nibabel as nib

        stack_nifti = nib.load(stack_path)
        stack_shape = stack_nifti.shape
        stack = stack_nifti.get_fdata()

        stack_voxel_spacing = np.diag(stack_nifti.affine)[:-1]

    return stack, stack_shape, stack_voxel_spacing
