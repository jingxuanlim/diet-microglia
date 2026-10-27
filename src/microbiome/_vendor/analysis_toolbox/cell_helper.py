# Vendored from analysis_toolbox (cell_helper.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from microbiome._vendor.analysis_toolbox.utils import rgetattr


def find_cell(
    cell_num,
    cmpnt_orig,
    cmpnt_wghts,
    image_mean,
    cell_volume=None,
    mask=1,
    pixbypix=False,
    z_offset=0,
    y_offset=0,
    x_offset=0,
):

    # to check for integers, floats, complex numbers,
    # Decimal and Fraction objects
    from numbers import Number

    if cell_volume is None:
        cell_volume = np.full(image_mean.shape, np.nan)

    # find origin
    origin = cmpnt_orig[cell_num]
    z, y, x = origin
    z -= z_offset
    y -= y_offset
    x -= x_offset

    # find location of non-zeros pixels
    py, px = np.where(cmpnt_wghts[cell_num])

    if pixbypix:

        for iy, ix in zip(py, px):

            if isinstance(mask, Number):
                cell_volume[int(z), int(y + iy), int(x + ix)] = mask
            else:
                cell_volume[int(z), int(y + iy), int(x + ix)] = cmpnt_wghts[cell_num][
                    int(iy), int(ix)
                ]

    else:

        if isinstance(mask, bool):
            cell_volume[
                int(z), (y + py).astype("int"), (x + px).astype("int")
            ] = cmpnt_wghts[cell_num][py.astype("int"), px.astype("int")]

        elif isinstance(mask, Number):
            cell_volume[int(z), (y + py).astype("int"),
                        (x + px).astype("int")] = mask

        else:  # if mask is False
            cell_volume[
                int(z), (y + py).astype("int"), (x + px).astype("int")
            ] = cmpnt_wghts[cell_num][py.astype("int"), px.astype("int")]

    return cell_volume


def find_cells(
    cell_list,
    cmpnt_orig,
    cmpnt_wghts,
    image_mean,
    cell_volume=None,
    mask=None,
    z_offset=0,
    y_offset=0,
    x_offset=0,
):

    if cell_volume is None:
        cell_volume = np.full(image_mean.shape, np.nan)

    # iterate over every single cell
    for icell, cell in enumerate(tqdm(cell_list)):

        if mask is not False:  # something or None

            # if mask is a list, with each element containing a number
            if (
                isinstance(mask, list)
                or isinstance(mask, np.ndarray)
                or isinstance(mask, pd.core.series.Series)
            ):

                cell_volume = find_cell(
                    cell,
                    cmpnt_orig,
                    cmpnt_wghts,
                    image_mean,
                    cell_volume=cell_volume,
                    mask=mask[icell],
                    pixbypix=False,
                    z_offset=z_offset,
                    y_offset=y_offset,
                    x_offset=x_offset,
                )

            # if mask is True, just use icell as the label for each cell
            # this has to be above int checking because bool is a subclass of int
            elif isinstance(mask, bool):
                cell_volume = find_cell(
                    cell,
                    cmpnt_orig,
                    cmpnt_wghts,
                    image_mean,
                    cell_volume=cell_volume,
                    mask=cell,
                    pixbypix=False,
                    z_offset=z_offset,
                    y_offset=y_offset,
                    x_offset=x_offset,
                )

            elif isinstance(mask, str) and mask == "weights":
                cell_volume = find_cell(
                    cell,
                    cmpnt_orig,
                    cmpnt_wghts,
                    image_mean,
                    cell_volume=cell_volume,
                    mask=False,
                    pixbypix=False,
                    z_offset=z_offset,
                    y_offset=y_offset,
                    x_offset=x_offset,
                )

            # if mask is an integer, all clusters will have the same color
            elif isinstance(mask, int) or isinstance(mask, float):
                cell_volume = find_cell(
                    cell,
                    cmpnt_orig,
                    cmpnt_wghts,
                    image_mean,
                    cell_volume=cell_volume,
                    mask=mask,
                    pixbypix=False,
                    z_offset=z_offset,
                    y_offset=y_offset,
                    x_offset=x_offset,
                )

            elif np.issubdtype(mask, np.integer):
                cell_volume = find_cell(
                    cell,
                    cmpnt_orig,
                    cmpnt_wghts,
                    image_mean,
                    cell_volume=cell_volume,
                    mask=mask,
                    pixbypix=False,
                    z_offset=z_offset,
                    y_offset=y_offset,
                    x_offset=x_offset,
                )

        else:
            # if mask is False, use cell sequence as label
            cell_volume = find_cell(
                cell,
                cmpnt_orig,
                cmpnt_wghts,
                image_mean,
                cell_volume=cell_volume,
                mask=icell,
                pixbypix=False,
                z_offset=z_offset,
                y_offset=y_offset,
                x_offset=x_offset,
            )

    return cell_volume


def plot_clusters_in_brain(
        spim_dset,
        im_mode,
        labels,
        cell_indices,
        included_cell_indices=None,
        individually=False,
        individual_labels=True,
        together=False,
        labels_to_plot=[],
        z_offset=0,
        y_offset=0,
        x_offset=0,
        cmap=None,
        make_plots=True,
        close_plots=False,
        **generate_filled_volume_kwargs
):
    """
    spim_dset: object with find_cells() method
    cell_indices: there are as many labels are there are cell_indices
    """

    subobjects = im_mode.split(".")

    if labels.ndim == 2:
        unique_labels = np.arange(len(labels_to_plot))
    elif labels.ndim == 1:
        unique_labels = np.unique(labels)

    if len(labels_to_plot) == 0:
        labels_to_plot = unique_labels

    if cmap is None:

        from microbiome._vendor.analysis_toolbox.colors import generate_random_colormap

        cmap = generate_random_colormap(
            len(unique_labels), fixed={0: np.array([1, 1, 1])}
        )

    # from microbiome._vendor.analysis_toolbox.visual import add_color_to_cmap
    # cmap = add_color_to_cmap("jet", 0, np.array([1, 1, 1, 1]))

    vols = {}

    from microbiome._vendor.analysis_toolbox.utils import operate_on_first_dim
    from microbiome._vendor.analysis_toolbox.utils import now_str

    if individually or (together and labels.ndim == 2):

        print(labels.ndim)
        if together:
            all_clusters_vol =\
                np.full(rgetattr(spim_dset, im_mode).volume_mean.shape, np.nan)

        for ilabel, label in enumerate(tqdm(labels_to_plot)):

            ## Determine: (1) cell_ids and (2) their labels
            if isinstance(label, (int, float)):

                 if individual_labels:
                     label_inds = label  # labels[operate_on_first_dim(labels, label)]
                     dict_label = label
                 else:
                     dict_label = label
                     label_inds = True

                 cell_inds = cell_indices[operate_on_first_dim(labels, label)]

            elif isinstance(label, list):

                if labels.ndim == 2:
                    from microbiome._vendor.analysis_toolbox.data.segments import return_cells_in_region
                    cell_idx = return_cells_in_region(labels, label)
                elif labels.ndim == 1:
                    cell_idx = np.concatenate([np.where(labels == lab)[0] for lab in label])

                cell_inds = cell_indices[cell_idx]

                if individual_labels:
                    dict_label = label
                    label_inds = ilabel
                else:
                    dict_label = label
                    label_inds = True

            if included_cell_indices is not None:
                cell_inds = np.intersect1d(cell_inds, included_cell_indices)

            if z_offset != 0:
                cluster_vol = spim_dset.generate_filled_volume(
                    label_inds,
                    im_mode,
                    valid_index=cell_inds,
                    z_offset=z_offset,
                    y_offset=y_offset,
                    x_offset=x_offset,
                    **generate_filled_volume_kwargs
                )

                if together:
                    all_clusters_vol = spim_dset.generate_filled_volume(
                        label_inds,
                        im_mode,
                        valid_index=cell_inds,
                        z_offset=z_offset,
                        y_offset=y_offset,
                        x_offset=x_offset,
                        volume=all_clusters_vol,
                        **generate_filled_volume_kwargs
                    )

            else:
                 cluster_vol = spim_dset.generate_filled_volume(
                    label_inds,
                    im_mode,
                    valid_index=cell_inds,
                     **generate_filled_volume_kwargs
                )

                 if together:
                     all_clusters_vol = spim_dset.generate_filled_volume(
                        label_inds,
                        im_mode,
                        valid_index=cell_inds,
                         volume=all_clusters_vol,
                         **generate_filled_volume_kwargs
                    )



            vols[str(dict_label)] = cluster_vol.astype("float16")

            if make_plots:

                proj_fig, proj_ax, gs = spim_dset.make_projections(
                    cluster_vol,
                    im_mode,
                    vmin=np.nanmin(unique_labels),
                    vmax=np.nanmax(unique_labels),
                    suptitle=f"Number of cells {len(cell_inds)}",
                    cmap=cmap,
                    compact=True,
                    # background=getattr(spim_dset, im_mode).volume_mean.astype("float32")[
                    # z_offset:, y_offset:, x_offset:
                    # ],
                    perspective=False,
                    savename=spim_dset.savepath + f"plot-clusters-in-brain_label-{label_inds}_{now_str()}.png",
                )

                if close_plots:
                    from microbiome._vendor.analysis_toolbox.visual import close_plot
                    close_plot(proj_fig)

        if together: vols["all"] = all_clusters_vol

    if together and labels.ndim == 1:

        clustered_vol = spim_dset.generate_filled_volume(
            labels, im_mode, valid_index=cell_indices
        )

        vols["all"] = clustered_vol

        if make_plots:
            from microbiome._vendor.analysis_toolbox.stacks_helper import make_projections

            # TODO:  update to spim_dset.make_projections()
            proj_fig, proj_ax, gs = spim_dset.make_projections(
                clustered_vol,
                im_mode,
                vmin=np.nanmin(labels),
                vmax=np.nanmax(labels),
                compact=True,
                cmap=cmap,
                perspective=False,
                max_p=True,
                savename=spim_dset.savepath + f"plot-clusters-in-brain_{now_str()}.png"
            )

    return vols
