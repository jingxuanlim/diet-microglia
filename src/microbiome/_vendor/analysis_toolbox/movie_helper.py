# Vendored from analysis_toolbox (movie_helper.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np
import matplotlib.pyplot as plt


def make_single_frame(t, make_single_frame_func=None, **kwargs):
    """
    generates frames: t increases for every frame (e.g. for
    fps=4Hz : [0,0.25,0.5,0.75,1.0, etc])
    """

    from moviepy.video.io.bindings import mplfig_to_npimage

    fps = kwargs['fps']
    max_t = kwargs['max_t']
    fig = kwargs['fig']

    index = np.round(t*fps).astype('int')  ## int(t*fps) doesn't work as it always rounds down

    if index < max_t:

        # plot_single_frame(ax, index, t_axis, units, tyx_vols, cmaps, alphas, vmins, vmaxs, figsize, aspect)
        make_single_frame_func(index=index, **kwargs)

    return mplfig_to_npimage(fig)


def plot_single_projection(ax, index, t_axis, units,
                           tyx_vols,
                           cmaps, alphas, vmins, vmaxs,
                           figsize, aspect, **kwargs):

    ax.clear()

    for tyx_vol, cmap, alpha, vmin, vmax in zip(tyx_vols, cmaps, alphas, vmins, vmaxs):
        ax.imshow(tyx_vol[index], cmap=cmap, aspect=aspect, alpha=alpha, vmin=vmin, vmax=vmax)

    if t_axis is not None: ax.text(0,0,f'{t_axis[index]:.2f} {units}',fontsize=6*figsize[0],color='white')

    ax.axis('off')


def plot_max_projections(
        fig, gs, xy_ax, xz_ax, yz_ax, cb_ax,
        index, t_axis, units, tyx_vols, cmaps, alphas, vmins, vmaxs, figsize,
        background, background_vmin, background_vmax, voxel_spacing, compact, rotate180,
        **kwargs):

    xy_ax.clear()
    xz_ax.clear()
    yz_ax.clear()
    cb_ax.clear()

    from microbiome._vendor.analysis_toolbox.stacks_helper import make_projections
    for tyx_vol, cmap, alpha, vmin, vmax in \
        zip(tyx_vols, cmaps, alphas, vmins, vmaxs):

        make_projections(tyx_vol[index],
                         cmap=cmap, alpha=alpha,
                         vmin=vmin, vmax=vmax, max_p=True,
                         voxel_spacing=voxel_spacing,
                         background=background, background_vmin=background_vmin, background_vmax=background_vmax, background_clahe=False,
                         rotate180=rotate180,
                         proj_fig=fig, gs=gs, xy_ax=xy_ax, xz_ax=xz_ax, yz_ax=yz_ax, cb_ax=cb_ax,
                         style='dark_background', compact=True, xlabel=r'$\Delta{}F/F$')

    if t_axis is not None: xy_ax.text(0,0,f'{t_axis[index]:.2f} {units}',fontsize=6*figsize[0],color='white')


def plot_single_var(
        fig, var_ax,
        single_vars,
        index, **kwargs):

    var_ax.clear()

    for single_var_name, single_var in single_vars.items():

        single_var_vals = np.unique(single_var)
        nonzero_single_var_vals = single_var_vals[single_var_vals != 0]

        if len(nonzero_single_var_vals) < 5:

            from microbiome._vendor.analysis_toolbox.colors import tableau_trafficlight9 as colorlist

            for i, val in enumerate(nonzero_single_var_vals):
                val_single_var = single_var == val

                var_ax.bar(x=0, height=val_single_var[index], width=np.pi*2, bottom=0, color=colorlist[i])
                var_ax.set_ylim([0, np.nanmax(val_single_var)])


        else:
            var_ax.bar(x=0, height=single_var[index], width=np.pi*2, bottom=0, color='white')
            var_ax.set_ylim([0, np.nanmax(single_var)])

        var_ax.set_title(single_var_name, color='white')

    var_ax.axis('off')


def plot_max_projections_and_single_var(
        fig, gs, xy_ax, xz_ax, yz_ax, var_ax, cb_ax,
        index, t_axis, units, tyx_vols, cmaps, alphas, vmins, vmaxs,
        voxel_spacing, compact, rotate180,
        single_vars,
        figsize,
        background, background_vmin, background_vmax,
        **kwargs):

    plot_max_projections(fig, gs, xy_ax, xz_ax, yz_ax, cb_ax,
                         index, t_axis, units,
                         tyx_vols,
                         cmaps, alphas, vmins, vmaxs,
                         figsize, background, background_vmin, background_vmax,
                         voxel_spacing, compact, rotate180)
    plot_single_var(fig, var_ax, single_vars, index)


def animate_volume(tyx_vols, single_vars=None,
                   voxel_spacing=(5, 0.403, 0.403),
                   compact=False, rotate180=False,
                   t_axis=None,
                   figsize=(6,6),
                   cmaps=None, alphas=None, vmins=None, vmaxs=None,
                   background=None, background_vmin=None, background_vmax=None,
                   aspect='equal', fps=1, units='s'):
    """
    Animate every slice you provide as part of a stack.

    tyx_vol [list]: list of tyx_vols
    fps [int]: determines number of frames per second
    aspect [float]: ratio of y:x or tyx_vol.shape[1]:tyx_vol.shape[2]
    """

    from moviepy.editor import VideoClip
    from functools import partial


    max_t = tyx_vols[0].shape[0]

    if tyx_vols[0].ndim == 3:

        fig, ax = plt.subplots(figsize=figsize, dpi=300, constrained_layout=True)
        make_frame = partial(make_single_frame,
                             make_single_frame_func=plot_single_projection,
                             fig=fig, ax=ax, figsize=figsize,
                             t_axis=t_axis, units=units, fps=fps, max_t=max_t,
                             tyx_vols=tyx_vols, cmaps=cmaps, alphas=alphas, vmins=vmins, vmaxs=vmaxs, aspect=aspect)

    elif tyx_vols[0].ndim == 4:

        from microbiome._vendor.analysis_toolbox.stacks_helper import create_projections_fig
        stack_dims = np.multiply(np.array(tyx_vols[0].shape[1:]), voxel_spacing)

        if single_vars is not None:

            print("Preparing to plot projections and variables...")

            fig, gs, xy_ax, xz_ax, yz_ax, var_ax, cb_ax = create_projections_fig(stack_dims=stack_dims,
                                                                                 dpi=300,
                                                                                 compact=compact,
                                                                                 single_var=True)

            make_frame = partial(make_single_frame,
                                 make_single_frame_func=plot_max_projections_and_single_var,
                                 fig=fig, gs=gs, xy_ax=xy_ax, xz_ax=xz_ax, yz_ax=yz_ax, var_ax=var_ax, cb_ax=cb_ax, figsize=figsize,
                                 t_axis=t_axis, units=units, fps=fps, max_t=max_t,
                                 tyx_vols=tyx_vols, cmaps=cmaps, alphas=alphas, vmins=vmins, vmaxs=vmaxs,
                                 aspect=aspect, voxel_spacing=voxel_spacing, compact=compact, rotate180=rotate180,
                                 single_vars=single_vars,
                                 background=background, background_vmin=background_vmin, background_vmax=background_vmax)

        else:

            print("Preparing to plot projections...")

            fig, gs, xy_ax, xz_ax, yz_ax, var_ax, cb_ax = create_projections_fig(stack_dims=stack_dims,
                                                                                 dpi=300,
                                                                                 compact=compact,
                                                                                 single_var=False)
            make_frame = partial(make_single_frame,
                                 make_single_frame_func=plot_max_projections,
                                 fig=fig, gs=gs, xy_ax=xy_ax, xz_ax=xz_ax, yz_ax=yz_ax, cb_ax=cb_ax, figsize=figsize,
                                 t_axis=t_axis, units=units, fps=fps, max_t=max_t,
                                 tyx_vols=tyx_vols, cmaps=cmaps, alphas=alphas, vmins=vmins, vmaxs=vmaxs,
                                 aspect=aspect, voxel_spacing=voxel_spacing, compact=compact, rotate180=rotate180,
                                 background=background, background_vmin=background_vmin, background_vmax=background_vmax)


    animation = VideoClip(make_frame, duration=max_t/fps)


    return animation


def calculate_figsize(width, tyx_vol, aspect_ratio):
    """
    width [float]: representing tyx_vol.shape[2]
    aspect_ratio [float]: voxel_spacing[2] / voxel_spacing[1]
    """

    height = width * (tyx_vol.shape[1]/tyx_vol.shape[2]) * aspect_ratio

    return (width, height)


def create_animation(tyx_vols, single_vars=None,
                     t_axis=None, units='s',
                     t_subsample_factor=1, t_subsample=None,
                     fps=1,
                     voxel_spacing=(5, 0.403, 0.403),
                     width=10, aspect_ratio=1,
                     compact=False, rotate180=False,
                     cmaps=None, alphas=None,
                     vmins=None, vmaxs=None,
                     vmin_percentiles=0, vmax_percentiles=100,
                     background=None, background_vmin=None, background_vmax=None,
                     savename=''):
    """
    tyx_vol [list]:
        list of either tyx or tzyx vols of the same size
        will work with numpy and zarr arrays

    single_vars [dict]:
        name of variable and array
    """

    plt.style.use('dark_background')
    from microbiome._vendor.analysis_toolbox.utils import check_array_shapes_identical
    num_vols = len(tyx_vols)
    if num_vols > 1: assert check_array_shapes_identical(tyx_vols)

    if t_subsample_factor > 1:
        tyx_vols = [tyx_vol[::t_subsample_factor] for tyx_vol in tyx_vols]
        if t_axis is not None: t_axis = t_axis[::t_subsample_factor]
    elif t_subsample_factor == 1 and t_subsample is not None:
        tyx_vols = [tyx_vol[np.isin(t_axis, t_subsample)] for tyx_vol in tyx_vols]
        if t_axis is not None: t_axis = t_axis[np.isin(t_axis, t_subsample)]

    from microbiome._vendor.analysis_toolbox.utils import is_numeric
    if is_numeric(vmin_percentiles): vmin_percentiles =  [vmin_percentiles] * num_vols
    if is_numeric(vmax_percentiles): vmax_percentiles = [vmax_percentiles] * num_vols
    if vmins is None: vmins = [np.nanpercentile(vol, percentile) for vol, percentile in zip(tyx_vols, vmin_percentiles)]
    if vmaxs is None: vmaxs = [np.nanpercentile(vol, percentile) for vol, percentile in zip(tyx_vols, vmax_percentiles)]
    if cmaps is None: cmaps = ['gray'] * num_vols
    if alphas is None: alphas = [1/num_vols] * num_vols

    figsize = calculate_figsize(width, tyx_vols[0], aspect_ratio)

    animation = animate_volume(tyx_vols,
                               single_vars=single_vars,
                               t_axis=t_axis,
                               voxel_spacing=voxel_spacing,
                               compact=compact,
                               rotate180=rotate180,
                               figsize=figsize,
                               vmins=vmins,
                               vmaxs=vmaxs,
                               cmaps=cmaps,
                               alphas=alphas,
                               aspect=aspect_ratio,
                               fps=fps,
                               units=units,
                               background=background,
                               background_vmin=background_vmin,
                               background_vmax=background_vmax
                               )

    # print(figsize, animation.size)  ## should be figsize * dpi

    plt.style.use('default')

    if savename:
        animation.write_videofile(savename,
                                  fps=fps,
                                  codec='h264',
                                  ffmpeg_params=[
                                      ## Chroma subsampling scheme
                                      '-pix_fmt', 'yuv420p',
                                      ## ensure that the width and height are divisible by 2
                                      ## https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
                                      ## (1) By padding
                                      '-vf',  'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                                      ## (2) By cropping
                                      # '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
                                      ## (3) By scaling, fixing height to 1280
                                      # '-vf', 'scale=1280:-2',
                                      ],
                                  logger='bar')
    else:
        return animation.ipython_display(fps=fps, loop=True, autoplay=True, progress_bar=False)
