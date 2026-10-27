# Vendored from analysis_toolbox (data/voluseg.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np
from tqdm.auto import tqdm


def calculate_dff(cell_timeseries, cell_baseline, background, offset=10):
    """
    Should be equivalent to Davis' version.
    See Davis' version in analysis_toolbox.preprocess_helper.mydff()
    In psuedo-code:
    data = cell_timeseries - background
    bl = baseline(data, window, percentile, downsample=downsample, axis=axis)
    return (data - bl) / (bl + baseline_offset)
    """
    return ((cell_timeseries - background) - (cell_baseline - background)) / (
        cell_baseline - background + offset
    )


def find_cell(
    cell_num,
    cell_z,
    cell_y,
    cell_x,
    cell_weights,
    stack_shape,
    cell_volume=None,
    mask=1,
    pixbypix=False,
    z_offset=0,
    y_offset=0,
    x_offset=0,
):
    """
    Adapted from mika_helper.py and made similar to cell_helper.find_cell()
    """

    z, y, x = stack_shape
    cell_z -= z_offset
    cell_y -= y_offset
    cell_x -= x_offset

    if cell_volume is None:
        cell_volume = np.full((z, y, x), np.nan)

    if pixbypix:
        for j in range(np.count_nonzero(cell_x[cell_num, :] >= 0)):

            if mask:
                cell_volume[
                    int(cell_z[cell_num, j]),
                    int(cell_y[cell_num, j]),
                    int(cell_x[cell_num, j]),
                ] = mask

            else:
                cell_volume[
                    int(cell_z[cell_num, j]),
                    int(cell_y[cell_num, j]),
                    int(cell_x[cell_num, j]),
                ] = cell_weights[cell_num, j]
    else:
        valid_cells = cell_x[cell_num] >= 0
        if mask:
            cell_volume[
                cell_z[cell_num, valid_cells],
                cell_y[cell_num, valid_cells],
                cell_x[cell_num, valid_cells],
            ] = mask

        else:
            cell_volume[
                cell_z[cell_num, valid_cells],
                cell_y[cell_num, valid_cells],
                cell_x[cell_num, valid_cells],
            ] = cell_weights[cell_num, valid_cells]

    return cell_volume


def find_cells_at_once(
    cell_list,
    cell_z,
    cell_y,
    cell_x,
    cell_weights,
    stack_shape=None,
    mask=1,
    cell_volume=None,
    z_offset=0,
    y_offset=0,
    x_offset=0,
):
    """
    mask:
      list or np.array: has to be the same size as cell_list
      int or float: same value for all in cell_list
      bool, specifically True: use segmented index (no meaning)
      'weights' [str]: use cell_weights, where all voxels in a cell have a different value
    """

    # Index using  cell_list
    if isinstance(mask, int):
        mask_values = np.ones_like(cell_list) * mask
    elif isinstance(mask, bool):
        if mask is True:
            mask_values = np.arange(len(cell_z))[cell_list]
        # TODO: what happens False?
    elif isinstance(mask, list):
        mask_values = np.array(mask)
    else:
        mask_values = mask

    cell_z = cell_z[cell_list]
    cell_y = cell_y[cell_list]
    cell_x = cell_x[cell_list]
    cell_weights = cell_weights[cell_list]

    # Index valid_cells
    valid_cells = cell_x >= 0

    cell_z = cell_z[valid_cells]
    cell_y = cell_y[valid_cells]
    cell_x = cell_x[valid_cells]
    cell_weights = cell_weights[valid_cells]

    if not isinstance(mask, str):
        from microbiome._vendor.analysis_toolbox.utils import num_to_nan

        mask_values = (
            num_to_nan(valid_cells.astype("float")) *
            mask_values[:, np.newaxis]
        )
        mask_values = mask_values[valid_cells]
    elif mask == "weights":
        mask_values = cell_weights

    cell_z -= z_offset
    cell_y -= y_offset
    cell_x -= x_offset

    if cell_volume is None:
        z, y, x = stack_shape
        cell_volume = np.full((z, y, x), np.nan)

    if mask_values.shape == cell_x.shape:
        cell_volume[cell_z, cell_y, cell_x] = mask_values
    else:
        cell_volume[cell_z, cell_y, cell_x] = cell_weights

    return cell_volume


def find_cells(
    cell_list,
    cell_z,
    cell_y,
    cell_x,
    cell_weights,
    stack_shape,
    cell_volume=None,
    mask=1,
    cellbycell=False,
    pixbypix=False,
    z_offset=0,
    y_offset=0,
    x_offset=0,
    debug=False,
):
    """
    Adapted from mika_helper.py and made similar to cell_helper.find_cell()
    """

    z, y, x = stack_shape

    if cell_volume is None:
        cell_volume = np.full((z, y, x), np.nan)

    if cellbycell:
        # iterate over every single cell
        for icell, cell in enumerate(tqdm(cell_list)):

            if mask is not False:  # something or None

                # if mask is a list, with each element containing a number
                if isinstance(mask, list) or isinstance(mask, np.ndarray):
                    cell_volume = find_cell(
                        cell,
                        cell_z,
                        cell_y,
                        cell_x,
                        cell_weights,
                        stack_shape,
                        cell_volume=cell_volume,
                        mask=mask[icell],
                        pixbypix=pixbypix,
                        z_offset=z_offset,
                        y_offset=y_offset,
                        x_offset=x_offset,
                    )
                # if mask is True, just use icell as the label for each cell
                # this has to be above int checking because bool is a subclass of int
                elif isinstance(mask, bool):

                    cell_volume = find_cell(
                        cell,
                        cell_z,
                        cell_y,
                        cell_x,
                        cell_weights,
                        stack_shape,
                        cell_volume=cell_volume,
                        mask=cell,
                        pixbypix=pixbypix,
                        z_offset=z_offset,
                        y_offset=y_offset,
                        x_offset=x_offset,
                    )

                elif isinstance(mask, str) and mask == "weights":
                    cell_volume = find_cell(
                        cell,
                        cell_z,
                        cell_y,
                        cell_x,
                        cell_weights,
                        stack_shape,
                        cell_volume=cell_volume,
                        mask=False,
                        pixbypix=pixbypix,
                        z_offset=z_offset,
                        y_offset=y_offset,
                        x_offset=x_offset,
                    )

                # if mask is an integer, all clusters will have the same color
                elif isinstance(mask, int) or isinstance(mask, float):
                    cell_volume = find_cell(
                        cell,
                        cell_z,
                        cell_y,
                        cell_x,
                        cell_weights,
                        stack_shape,
                        cell_volume=cell_volume,
                        mask=mask,
                        pixbypix=pixbypix,
                        z_offset=z_offset,
                        y_offset=y_offset,
                        x_offset=x_offset,
                    )

                elif np.issubdtype(mask, np.integer):
                    cell_volume = find_cell(
                        cell,
                        cell_z,
                        cell_y,
                        cell_x,
                        cell_weights,
                        stack_shape,
                        cell_volume=cell_volume,
                        mask=mask,
                        pixbypix=pixbypix,
                        z_offset=z_offset,
                        y_offset=y_offset,
                        x_offset=x_offset,
                    )

            else:
                # if mask is False, use cell sequence as label
                cell_volume = find_cell(
                    cell,
                    cell_z,
                    cell_y,
                    cell_x,
                    cell_weights,
                    stack_shape,
                    cell_volume=cell_volume,
                    mask=icell,
                    pixbypix=pixbypix,
                    z_offset=z_offset,
                    y_offset=y_offset,
                    x_offset=x_offset,
                )
    else:

        cell_volume = find_cells_at_once(
            cell_list,
            cell_z,
            cell_y,
            cell_x,
            cell_weights,
            cell_volume=cell_volume,
            mask=mask,
            z_offset=z_offset,
            y_offset=y_offset,
            x_offset=x_offset,
        )

    return cell_volume
