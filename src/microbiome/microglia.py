import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile

# Fast local scratch space for copies of raw data; override with $MICROBIOME_TEMP_DIR.
TEMP_DIR = os.environ.get("MICROBIOME_TEMP_DIR", tempfile.gettempdir())


from collections import OrderedDict
condition_colors = OrderedDict()
condition_colors['unfed'] =  'black'
condition_colors['Sparos'] = 'tab:red'
condition_colors['para'] = 'tab:green'
condition_colors['yolk'] = 'gold'
condition_colors['yolk+DMSO'] = 'darkkhaki'
condition_colors['yolk+SB225002'] = 'tab:orange'
condition_colors['PC'] =  'blue'
condition_colors['PA'] =  'tab:grey'
condition_colors['CLB'] =  'navy'
condition_colors['PC+DMSO'] = 'darkblue'
condition_colors['PC+SB225002'] = 'tab:cyan'
condition_colors['EYPC'] = 'tab:blue'
condition_colors['LPC'] = 'tab:purple'

manuscript_colors = OrderedDict()
manuscript_colors['unfed'] =  (0, 0, 0)
manuscript_colors['Sparos'] = 'tab:grey'
manuscript_colors['yolk'] = (150, 117, 16) ## goldish
manuscript_colors['yolk+DMSO'] = 'gold'
manuscript_colors['yolk+SB225002'] = 'khaki'
manuscript_colors['PC'] =  (66, 94, 26)  ## greenish
manuscript_colors['PA'] =  (148, 100, 31)
manuscript_colors['CLB'] =  'navy'
manuscript_colors['PC+DMSO'] = 'green'
manuscript_colors['PC+SB225002'] = 'tab:cyan'
manuscript_colors['EYPC'] = (66, 94, 26)  ## greenish
manuscript_colors['LPC'] = 'tab:green'
manuscript_colors['cholesterol'] = (255, 70, 70)

def extract_objects(im_stack: np.array, debug: bool = False):
    """
    Extract objects from a 3D image stack using the Li thresholding method.
    Parameters
    ----------
    im_stack : np.array
        3D image stack to be processed.
    debug : bool
        If True, display debug information.
    Returns
    -------
    li_thresholded : np.array
        Processed image stack with objects extracted.
    """
    import scipy.ndimage as ndi
    import skimage as ski

    # Apply median filter to denoise the image stack        
    denoised = ndi.median_filter(im_stack, size=3)

    li_thresholded = denoised > ski.filters.threshold_li(denoised)

    return li_thresholded

def extract_sparse_cells(
        im_stack: np.array, 
        spacing: tuple,
        erode_before_removing_large_objects: bool = False,
        width: int = 35,
        debug: bool = True
        ) -> np.array:
    """
    Extracts sparse cells from a 3D image stack assuming that the skin objects are bigger than the cells.
    
    Parameters:
    - im_stack: 3D numpy array representing the image stack.
    - spacing: List of spacing values for each dimension.
    
    Returns:
    - small_obj_labels: 3D label array of the small objects.
    """

    import skimage as ski
    from skimage.morphology import octahedron
    from scipy import ndimage as ndi

    if erode_before_removing_large_objects:

        eroded = ski.morphology.binary_erosion(im_stack, footprint=None)

        remove_objects = ski.morphology.remove_small_objects(
            eroded, 
            min_size=width ** 3
        )

        small_obj_removed = ski.morphology.binary_dilation(remove_objects, footprint=None)

    else:

        small_obj_removed = ski.morphology.remove_small_objects(im_stack, min_size=width ** 3)

    subtracted = (im_stack.astype('int') - small_obj_removed.astype('int')).astype('bool')
    small_obj_dilated = ski.morphology.binary_dilation(subtracted, footprint=octahedron(3))

    large_obj_labels = ski.measure.label(small_obj_removed)
    small_obj_labels = ski.measure.label(small_obj_dilated)

    if debug:

        from microbiome._vendor.analysis_toolbox.colors import generate_random_colormap
        large_obj_cmap = generate_random_colormap(np.nanmax(large_obj_labels))
        small_obj_cmap = generate_random_colormap(np.nanmax(small_obj_labels))

        from microbiome._vendor.analysis_toolbox.stacks_helper import stack_comparison_long

        stack_comparison_long([im_stack, large_obj_labels, small_obj_labels, im_stack], 
                            titles=['raw', f'large object labels (n={np.nanmax(large_obj_labels)})', f'small object labels (n={np.nanmax(small_obj_labels)})', 'raw'],
                            vmins=[None, np.nanmin(large_obj_labels), np.nanmin(small_obj_labels), None,],
                            vmaxs=[None, np.nanmax(large_obj_labels), np.nanmax(small_obj_labels), None],
                            color_list=["gray", large_obj_cmap, small_obj_cmap, "gray"], 
                            figsize_scale=10, 
                            show_engine='imshow')

    return small_obj_labels

def extract_microglia(
        im_stack: np.array, 
        spacing: tuple, 
        erode_before_removing_large_objects: bool = False,
        width: int = 35,
        debug: bool = False
        ) -> np.array:
    """
    Extracts microglia from a 3D image stack.
    
    Parameters:
    - im_stack: 3D numpy array representing the image stack.
    - spacing: List of spacing values for each dimension.
    
    Returns:
    - small_obj_labels: 3D label array of the small objects.
    """
    if im_stack.ndim == 4:
        dim_reduced = True
        im_stack = np.squeeze(im_stack)
    else:
        dim_reduced = False

    li_thresholded = extract_objects(im_stack, debug=debug)
    small_obj_labels = extract_sparse_cells(
        li_thresholded, 
        spacing, 
        erode_before_removing_large_objects=erode_before_removing_large_objects, 
        width=width, 
        debug=debug
    )

    if dim_reduced:
        return small_obj_labels[np.newaxis,...]
    else:
        return small_obj_labels

def extract_microglia_from_stack(
        image_stack, 
        vox_spacing, 
        small_obj_labels_path, 
        copy_data_to_local=True,
        image_stack_path='', 
        compute_mode='dask',
        dask_client_compute_kwargs={}
        ):

    from microbiome.microglia import extract_microglia
    from microbiome._vendor.analysis_toolbox.parallel import dask_client_compute
    from microbiome._vendor.analysis_toolbox.utils import change_root_dir_in_path
    import dask.array as da
    import shutil

    if compute_mode == 'dask':

        if copy_data_to_local:

            temp_local_path = change_root_dir_in_path(image_stack_path, TEMP_DIR)

            if os.path.exists(temp_local_path):
                print(f"Temp local path {temp_local_path} already exists. Removing...")
                if os.path.isdir(temp_local_path):
                    shutil.rmtree(temp_local_path)
                else:
                    os.remove(temp_local_path)

            if not os.path.exists(temp_local_path):
                # copy directory
                shutil.copytree(image_stack_path, temp_local_path)


            data_to_extract_from = da.from_zarr(temp_local_path, inline_array=True)

        else:

            data_to_extract_from = image_stack


        small_obj_labels_da = data_to_extract_from.map_blocks(extract_microglia, 
                                                              spacing=vox_spacing, 
                                                              meta=np.array((), dtype='int32'),)

        display(small_obj_labels_da)

        dask_client_compute(small_obj_labels_da, 
                            zarr_path=small_obj_labels_path,
                            **dask_client_compute_kwargs
                            )

        small_obj_labels = da.from_zarr(small_obj_labels_path, inline_array=True)


        if copy_data_to_local:
            os.remove(temp_local_path)


    ## INCOMPLETE
    elif compute_mode == 'xarray':

        small_obj_labels = spim_dset.im_CM0_raw.data.chunk({'T': 1, 'Z': 101, 'Y': 512, 'X': 512}).map_blocks(extract_microglia, 
                                                                                                                        kwargs={"spacing": vox_spacing}, 
                                                                                                                        template=spim_dset.im_CM0_raw.data.chunk({'T': 1, 'Z': 101, 'Y': 512, 'X': 512}))

        display(small_obj_labels)

    return small_obj_labels


def compute_compactness(area, perimeter):

    if perimeter == 0:
        return np.nan
    else:
        return (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

def format_regionprops_table(df, timepoint, expt_id):

    df['time_s'] = timepoint
    df['expt'] = expt_id

def rechunk_nikon_twophoton_data(nikon_data, nikon_data_path, rechunked_data_path, copy_data_to_local=False):

    from microbiome._vendor.analysis_toolbox.utils import change_root_dir_in_path
    from microbiome._vendor.analysis_toolbox.parallel import dask_client_compute

    import shutil
    import nd2
    import os
    
    if copy_data_to_local:

        temp_local_path = change_root_dir_in_path(nikon_data_path, TEMP_DIR)

        if os.path.exists(temp_local_path):

            os.remove(temp_local_path)

        if not os.path.exists(temp_local_path):

            shutil.copy2(nikon_data_path, temp_local_path)

        data_local = nd2.imread(temp_local_path, xarray=True, dask=True)

        data_to_process = data_local.data
    else:
        data_to_process = nikon_data.data

    from rechunker import rechunk
    worker_max_mem = 40//2
    rechunked = rechunk(data_to_process, 
                        target_chunks=(1, data_to_process.shape[1], data_to_process.shape[2], data_to_process.shape[3]), 
                        target_store=rechunked_data_path,
                        max_mem=f"{worker_max_mem/2}GB",
                        temp_store=os.path.join(TEMP_DIR, 'rechunker_intermediate'))

    dask_client_compute(
        rechunked,
        func="execute",
        num_workers=2,
        num_cores_per_worker=1,
        max_task_size=worker_max_mem,
    )

    if os.path.exists(os.path.join(TEMP_DIR, 'rechunker_intermediate')):
        shutil.rmtree(os.path.join(TEMP_DIR, 'rechunker_intermediate'))

    if copy_data_to_local:
        del data_to_process
        os.remove(temp_local_path)

def characterize_extracted_microglia(small_obj_labels, spim_dset, small_objs_df_path, baseline_small_obj_labels=None):

    import skimage as ski

    small_objs_list = []
    nrows_counter = 0

    # - area, solidity and extent are the holy trinity
    # - however, my images are overexposed, so it fills in the gaps between microglia branches, 
    #   making them look like solid blobs
    # - I think feret_diameter_max could be useful
    properties = [
        'label', 
        'area', 
        'centroid', 
        'coords', 
        'image', 
        'axis_major_length',
        # 'feret_diameter_max', ## if your objects are large (like a brain outline), the number of points involved in that calculation can explode
        # 'solidity',  # Solidity: how "holey" or "branched" the object is
        # 'extent',  # Extent: how much of the bounding box is filled by the object
        ]

    if baseline_small_obj_labels is not None:

        small_objs_dict = ski.measure.regionprops_table(
            baseline_small_obj_labels.compute(scheduler='synchronous'),
            properties=properties
            )

        small_objs_df = pd.DataFrame(small_objs_dict)

        format_regionprops_table(
            small_objs_df, 
            timepoint = -15 * 60,   ## 15 mins before
            expt_id=f"{spim_dset.expt_date.strftime('%Y%m%d')} {spim_dset.fish_num}"
            )  

        nrows_timepoint = len(small_objs_df)
        # print(nrows_timepoint)
        nrows_counter += nrows_timepoint 

        small_objs_list.append(small_objs_df)

    for t, labels in tqdm(zip(spim_dset.im_CM0_raw.t_axis_s, small_obj_labels), total=len(spim_dset.im_CM0_raw.t_axis_s)):

        print(f"Processing time point {t} seconds / {t/3600} hours...")

        small_objs_dict = ski.measure.regionprops_table(labels.compute(scheduler='synchronous'),
                                                        properties=properties
                                                        )

        
        small_objs_df = pd.DataFrame(small_objs_dict)

        format_regionprops_table(
            small_objs_df, 
            timepoint = t,   ## current time point
            expt_id=f"{spim_dset.expt_date.strftime('%Y%m%d')} {spim_dset.fish_num}"
            )  

        nrows_timepoint = len(small_objs_df)
        print(nrows_timepoint)
        nrows_counter += nrows_timepoint 

        small_objs_list.append(small_objs_df)

    small_objs_df = pd.concat(small_objs_list, ignore_index=True)
    assert nrows_counter == len(small_objs_df)

    display(small_objs_df)
    small_objs_df.to_pickle(small_objs_df_path)

    return small_objs_df


def extract_or_load_microglia_labels(
        spim_dset,
        recompute=False,
        compute_mode = 'dask',
        num_workers = 12, 
        num_cores_per_worker = 1,
        copy_data_to_local = True,
        save_memory = True,
        ):

    from microbiome._vendor.analysis_toolbox.fileio import read_zarr
    import dask.array as da
    from microbiome.microglia import extract_microglia_from_stack
    from microbiome.microglia import rechunk_nikon_twophoton_data
    import shutil
    import nd2

    print(f"Processing dataset: {spim_dset.expt_date.strftime('%Y%m%d')} {spim_dset.fish_num}...")

    spim_dset.im_CM0_raw_rechunked_path = f'{spim_dset.procpath}im_CM0_raw_rechunked.zarr'
    spim_dset.small_obj_labels_path = f'{spim_dset.procpath}small_obj_labels.zarr'

    if recompute:
        if os.path.exists(spim_dset.small_obj_labels_path):
            print("small_obj_labels already exists. Removing and recomputing...")
            shutil.rmtree(spim_dset.small_obj_labels_path)

    if not os.path.exists(spim_dset.small_obj_labels_path):

        print("small_obj_labels doesn't exist. Computing...")

        if not os.path.exists(spim_dset.im_CM0_raw_rechunked_path):

            rechunk_nikon_twophoton_data(
                spim_dset.im_CM0_raw.data, 
                spim_dset.im_CM0_raw.path, 
                spim_dset.im_CM0_raw_rechunked_path, 
                copy_data_to_local=copy_data_to_local
                )

        spim_dset.im_CM0_raw_rechunked = da.from_zarr(
            spim_dset.im_CM0_raw_rechunked_path, 
            inline_array=True
            )

        display(spim_dset.im_CM0_raw_rechunked)

        dask_client_compute_kwargs = {
            'num_workers': num_workers,
            'num_cores_per_worker': num_cores_per_worker,
        }

        spim_dset.small_obj_labels = extract_microglia_from_stack(
            spim_dset.im_CM0_raw_rechunked, 
            spim_dset.im_CM0_raw.vox_spacing, 
            spim_dset.small_obj_labels_path, 
            copy_data_to_local=copy_data_to_local, 
            image_stack_path=spim_dset.im_CM0_raw_rechunked_path, 
            compute_mode='dask',
            dask_client_compute_kwargs=dask_client_compute_kwargs
            )

    else:
        print("small_obj_labels already exists. Loading from disk...")
        spim_dset.small_obj_labels = da.from_zarr(spim_dset.small_obj_labels_path, inline_array=True)
        display(spim_dset.small_obj_labels)


def extract_or_load_microglia_characterization(
        spim_dset,
        overwrite=False
):

    from microbiome._vendor.analysis_toolbox.fileio import remove_path
    from microbiome.microglia import characterize_extracted_microglia

    print(f"Processing dataset: {spim_dset.expt_date.strftime('%Y%m%d')} {spim_dset.fish_num}...")
    spim_dset.small_objs_df_path = f'{spim_dset.procpath}small_obj_df.pkl'

    if overwrite: 
        if os.path.exists(spim_dset.small_objs_df_path):
            remove_path(spim_dset.small_objs_df_path, safe=False)

    if not os.path.exists(spim_dset.small_objs_df_path):

        print("small_objs_df doesn't exist. Computing...")

        if hasattr(spim_dset, 'baseline_spim_dset'): baseline_small_obj_labels = spim_dset.baseline_spim_dset.small_obj_labels
        else: baseline_small_obj_labels = None

        spim_dset.small_objs_df = characterize_extracted_microglia(spim_dset.small_obj_labels, spim_dset, spim_dset.small_objs_df_path, baseline_small_obj_labels=baseline_small_obj_labels)
            
    else:
        
        print("small_objs_df already exists. Loading from disk...")
        spim_dset.small_objs_df = pd.read_pickle(spim_dset.small_objs_df_path)
        display(spim_dset.small_objs_df)

def generate_or_load_microglia_stack(
        spim_dset, 
        recompute=False, 
        copy_data_to_local=True,
        brain_area_lower=5000, 
        brain_axis_major_length_lower=100, 
        microglia_area_lower=100,
        microglia_area_upper=5000, 
        microglia_axis_major_length_upper=50,
        plot_microglia_stack=False
        ):

    from microbiome.microglia import mask_and_plot, apply_func_to_column_and_find_closest, make_masked_stack_from_regions
    from microbiome._vendor.analysis_toolbox.fileio import load_gsheet, fill_zarr_along_axis0
    from microbiome._vendor.analysis_toolbox.utils import now_str, change_root_dir_in_path
    import dask.array as da

    spim_dset.microglia_stack_path = f'{spim_dset.procpath}microglia_stack.zarr'

    if recompute:
        if os.path.exists(spim_dset.microglia_stack_path):
            print("microglia_stack already exists. Removing and recomputing...")
            shutil.rmtree(spim_dset.microglia_stack_path)

    if not os.path.exists(spim_dset.microglia_stack_path):

        if copy_data_to_local:

            temp_local_path = change_root_dir_in_path(spim_dset.im_CM0_raw.path, TEMP_DIR)

            if not os.path.exists(temp_local_path):
                shutil.copy2(spim_dset.im_CM0_raw.path, temp_local_path)

            spim_dset.im_CM0_raw_local = nd2.imread(temp_local_path, xarray=True, dask=True)

            data_to_process = spim_dset.im_CM0_raw_local.data
        else:
            data_to_process = spim_dset.im_CM0_raw.data.data

        fig_datetime = now_str()

        timepoints = spim_dset.small_objs_df['time_h'].unique()       
    
        for t in tqdm(spim_dset.small_objs_df['time_h'].unique(), position=1):

            if t in timepoints:

                if t < 0:

                    data_volume = spim_dset.baseline_spim_dset.im_CM0_raw.data.data

                else:

                    indices = np.where(spim_dset.im_CM0_raw.t_axis_h == t)[0]
                    assert len(indices) == 1
                    i = indices[0]
                    data_volume = data_to_process[i]

                # Huge, solid, and occupies a large portion of its bounding box.
                brain_query = (spim_dset.small_objs_df['area'] > brain_area_lower) & (spim_dset.small_objs_df['axis_major_length'] > brain_axis_major_length_lower) ## & (spim_dset.small_objs_df['solidity'] > brain_solidity_lower) 

                # 3. Define the Microglia Filter (The "Dual-State" Logic)
                # We use Area as the primary gate, then check for shape characteristics.
                microglia_query = (spim_dset.small_objs_df['area'] > microglia_area_lower) & (spim_dset.small_objs_df['axis_major_length'] < microglia_area_upper) & (spim_dset.small_objs_df['axis_major_length'] < microglia_axis_major_length_upper)


                selected_data = spim_dset.small_objs_df.loc[np.logical_and.reduce([
                    spim_dset.small_objs_df['time_h'] == t, 
                    microglia_query, 
                    np.logical_not(brain_query)
                ])]

                print(f"Number of microglia candidates at {t} h: {len(selected_data)}")

                
                microglia_stack = make_masked_stack_from_regions(spim_dset, selected_data, data_volume)
                fill_zarr_along_axis0(spim_dset.microglia_stack_path, spim_dset.im_CM0_raw.data.data.shape, i, microglia_stack)

                if plot_microglia_stack:
                    mask_and_plot(spim_dset, 
                                  selected_data,
                                  data_volume, 
                                  suptitle=f'{t:.2f} h',
                                  savename=[f'{spim_dset.savepath}small-objs_t-{t:.2f}_{fig_datetime}.png', 
                                            f'{spim_dset.savepath}small-objs_t-{t:.2f}_{fig_datetime}.svg']
                                            )

    return da.from_zarr(spim_dset.microglia_stack_path, inline_array=True)            

def generate_or_load_outline_stack(spim_dset, recompute=False, copy_data_to_local=True):

    from microbiome._vendor.analysis_toolbox.utils import now_str, change_root_dir_in_path
    from microbiome._vendor.analysis_toolbox.parallel import dask_client_compute
    import shutil

    spim_dset.brain_outline_path = f'{spim_dset.procpath}brain_outline.zarr'

    if recompute:
        if os.path.exists(spim_dset.brain_outline_path):
            print("brain_outline.zarr already exists. Removing...")
            shutil.rmtree(spim_dset.brain_outline_path)

    if not os.path.exists(spim_dset.brain_outline_path):

        if copy_data_to_local:

            temp_local_path = change_root_dir_in_path(spim_dset.im_CM0_raw_rechunked_path, TEMP_DIR)

            if os.path.exists(temp_local_path):
                print(f"Temp local path {temp_local_path} already exists. Removing...")
                if os.path.isdir(temp_local_path):
                    shutil.rmtree(temp_local_path)
                else:
                    os.remove(temp_local_path)

            if not os.path.exists(temp_local_path):
                # copy directory
                shutil.copytree(spim_dset.im_CM0_raw_rechunked_path, temp_local_path)

            data_to_extract_from = da.from_zarr(temp_local_path, inline_array=True)

        else:

            data_to_extract_from = spim_dset.im_CM0_raw_rechunked


        brain_outline = (data_to_extract_from - da.nan_to_num(spim_dset.microglia_stack)).astype(spim_dset.im_CM0_raw_rechunked.dtype)
        display(spim_dset.brain_outline)           

        num_workers = 2
        worker_max_mem = 32//num_workers

        dask_client_compute(
            brain_outline,
            zarr_path=f'{spim_dset.procpath}brain_outline.zarr',
            num_workers=num_workers,
            num_cores_per_worker=1,
            max_task_size=worker_max_mem,
        )


def apply_func_to_column_and_find_closest(df, func, column_name, n=5, return_indices=False):

    val = getattr(df[column_name], func)()
    indices = (df[column_name] - val).abs().nsmallest(n).index
    closest = df.loc[indices] 

    if return_indices:
        return closest, indices
    else:
        return closest
    

def put_in_center(big_array, small_array):
    """
    Puts a smaller n-dimensional numpy array into the center of a larger 
    n-dimensional numpy array.
    
    Args:
        big_array (np.ndarray): The larger array (modified in place).
        small_array (np.ndarray): The smaller array to place in the center.
    
    Raises:
        ValueError: If arrays have different dimensions or small array is larger.
    """
    if big_array.ndim != small_array.ndim:
        raise ValueError("Arrays must have the same number of dimensions")
        
    for i in range(big_array.ndim):
        if small_array.shape[i] > big_array.shape[i]:
            raise ValueError("Small array must be smaller than big array in all dimensions")

    # Calculate the starting and ending indices for each dimension
    start_indices = [(big_array.shape[i] - small_array.shape[i]) // 2 for i in range(big_array.ndim)]
    end_indices = [start_indices[i] + small_array.shape[i] for i in range(big_array.ndim)]

    # Create a slice object for indexing
    # This dynamically builds the multidimensional slice, e.g., big_array[start_0:end_0, start_1:end_1, ...]
    slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))

    # Assign the small array to the center of the big array
    big_array[slices] = small_array

    return big_array


def make_masked_stack_from_regions(spim_dset, cell_df, data_volume, return_cell_coords=False):
    """
    Generates a "masked stack" (3D volume) containing only the voxels belonging to 
    specified microglia cells, extracted from a larger data volume.

    This function is particularly useful for creating visualization crops (e.g., 64x64x64 or 128x128x128)
    centered on each microglia cell for analysis of cell morphology.

    Args:
        spim_dset: An object containing experimental data, specifically expected to 
                   have a method `generate_filled_volume` that can be used to create 
                   a boolean mask for the cell coordinates.
        cell_df (pd.DataFrame): DataFrame containing the coordinates of the cells to 
                                be extracted. It must have a 'coords' column, where 
                                each entry is a list or tuple of (x, y, z) indices.
        data_volume (da.Array or np.ndarray): The source 3D data volume from which to 
                                               extract the microglia. It can be a 
                                               dask array or a numpy array.
        return_cell_coords (bool, optional): If True, the function will also return the 
                                             list of coordinates corresponding to the 
                                             extracted cell(s). Defaults to False.

    Returns:
        np.ndarray: A 3D numpy array (the "masked stack") containing the intensity values 
                    of the voxels belonging to the specified cell(s) from the 
                    `data_volume`. If `return_cell_coords` is True, it returns a tuple 
                    (cell_vol, cell_coords).

    Raises:
        ValueError: If `spim_dset` does not have the `generate_filled_volume` method.

    Example:
        >>> # Assuming 'ds' is a valid spim_dset object with small_obj_labels
        >>> # Extract the first 10 cells
        >>> cells_to_extract = ds.small_objs_df.head(10)
        >>> 
        >>> # Generate masked stacks (e.g., 64x64x64 crops centered on cells)
        >>> masked_stacks = make_masked_stack_from_regions(
        >>>     ds, 
        >>>     cells_to_extract, 
        >>>     ds.im_CM0_raw.data,
        >>>     return_cell_coords=True
        >>> )
        >>> 
        >>> # masked_stacks[0] is the 3D volume for the first cell.
        >>> # masked_stacks[1] is the coordinate list for the first cell.
        
    """

    from microbiome._vendor.analysis_toolbox.stacks_helper import convert_to_list_of_coords
    import skimage as ski

    cell_coords = convert_to_list_of_coords(cell_df['coords'])

    cell_vol_bool = spim_dset.generate_filled_volume(True, 'im_CM0_raw', valid_index=cell_coords)
    cell_vol_bool_dilated = ski.morphology.binary_dilation(cell_vol_bool, footprint=None)

    cell_vol = np.multiply(\
        data_volume,
        cell_vol_bool
        ).compute(scheduler='synchronous')

    if return_cell_coords:
        return cell_vol, cell_coords
    else:
        return cell_vol


def plot_masked_projection(
        spim_dset, 
        cell_vol,
        zoom=None, 
        cell_df=None,
        cell_coords=None, 
        suptitle='', 
        savename=None,
        return_masked_volume=False
        ):
    """
    """

    import skimage as ski

    if zoom is not None:

        if len(cell_df) == 1:
            volume = cell_df.iloc[0]['area'] * np.multiply.reduce(spim_dset.im_CM0_raw.vox_spacing)   
            suptitle = suptitle + f' volume={volume:.0f} ' + r'$\mu$m$^3$'

        z_median = int(np.median(cell_coords[0]))
        y_median = int(np.median(cell_coords[1]))
        x_median = int(np.median(cell_coords[2]))

        zoomed_cell_vol = np.full(zoom, np.nan)

        data_vol = cell_vol[np.clip(z_median-zoom[0]//2, a_min=0, a_max=None):np.clip(z_median+zoom[0]//2, a_min=None, a_max=cell_vol.shape[0]),
                            np.clip(y_median-zoom[1]//2, a_min=0, a_max=None):np.clip(y_median+zoom[1]//2, a_min=None, a_max=cell_vol.shape[1]),
                            np.clip(x_median-zoom[2]//2, a_min=0, a_max=None):np.clip(x_median+zoom[2]//2, a_min=None, a_max=cell_vol.shape[2])]

        # put the smaller zoomed data into the center of the full zoom size array
        cell_vol = put_in_center(zoomed_cell_vol, data_vol)

    masked_vol = np.nan_to_num(cell_vol)

    proj_fig, proj_ax, gs = spim_dset.make_projections(masked_vol, 
                            'im_CM0_raw', 
                            background=None,
                            cmap='grey', 
                            compact=0.02,
                            suptitle=suptitle,
                            savename=savename)

    if return_masked_volume:
        return masked_vol   

def mask_and_plot(
        spim_dset,
        cell_df, 
        data_volume,
        zoom=None,
        suptitle='', 
        savename=None, 
        return_masked_volume=False
        ):

    cell_vol, cell_coords = make_masked_stack_from_regions(spim_dset, cell_df, data_volume, return_cell_coords=True)

    plot_masked_projection(spim_dset, cell_vol, zoom=zoom, cell_df=cell_df, cell_coords=cell_coords, suptitle=suptitle, savename=savename)


def filter_microglia(
    mask_3d: np.ndarray, 
    spacing: tuple, 
    brain_volume_lower_um3: float, 
    brain_axis_major_length_lower_um: float,
    max_dilation_um: float = 1.0,
    max_bridge_um: float = 2.0,
    uniform_dilation_towards_um: float = 0.5,
    uniform_dilation_away_um: float = 1.0,
    debug: bool = False,
    plot_func: callable = None,
    plot_kwargs: dict = None
) -> tuple:
    """
    Filter microglia candidates from brain outlines in a 3D ZYX stack.
    All shape descriptors are physically scaled to handle anisotropic imaging.
    Dilation is spatially adaptive, allowing higher dilation away from the image center.
    
    Parameters
    ----------
    mask_3d : np.ndarray
        3D binary or boolean image stack of segmented objects (shape: Z x Y x X).
    spacing : tuple of floats
        Voxel spacing in micrometers in ZYX sequence: (dz, dy, dx).
    brain_volume_lower_um3 : float
        The minimum physical volume in cubic micrometers (um3) to consider an object a brain/skin segment.
    brain_axis_major_length_lower_um : float
        The minimum physical major axis length in micrometers (um) to consider an object a brain/skin segment.
    max_dilation_um : float, optional
        The maximum distance (in micrometers) to dilate the brain segments towards each other. Default is 1.0.
    max_bridge_um : float, optional
        The maximum edge-to-edge distance (in micrometers) between two brain segments 
        for them to be considered close enough to connect. Default is 2.0.
    uniform_dilation_towards_um : float, optional
        Dilation limit (in micrometers) for expansion towards the image center. Default is 0.5.
    uniform_dilation_away_um : float, optional
        Dilation limit (in micrometers) for expansion away from the image center. Default is 1.0.
    debug : bool, optional
        If True, plot the results. Default is False.
    plot_func : callable, optional
        Function to plot 3D stacks. E.g. spim_dset.make_projections. Default is None.
    plot_kwargs : dict, optional
        Dictionary of keyword arguments passed to plot_func. Default is None.
        
    Returns
    -------
    final_brain_mask : np.ndarray
        3D boolean mask containing the brain outlines and strictly absorbed segments.
    final_microglia_candidates : np.ndarray
        3D boolean mask of the remaining potential microglia candidates.
    """
    import numpy as np
    from scipy import ndimage
    import pandas as pd
    from skimage.measure import label, regionprops_table
    import time

    # 0. Extract objects
    mask_3d_thresholded = extract_objects(mask_3d)
    
    # 1. Separate objects by physical size and shape properties using spacing=spacing
    print(f'Segmenting the 3D mask into individual objects...')
    start = time.time()
    label_image = label(mask_3d_thresholded)
    end = time.time()
    print(f'Segmenting the 3D mask into individual objects took {end - start} seconds.')
    
    print(f'Characterizing individual objects by physical size and shape properties...')
    start = time.time()
    props = pd.DataFrame(
        regionprops_table(label_image, properties=('label', 'area', 'axis_major_length'), spacing=spacing)
    )
    end = time.time()
    print(f'Characterizing individual objects by physical size and shape properties took {end - start} seconds.')
    
    # Rename columns to reflect physical units
    props = props.rename(columns={'area': 'volume_um3', 'axis_major_length': 'axis_major_length_um'})
    
    # Identify the 'Big' objects (Brain/Skin) using physical volume and length thresholds
    brain_query = (props['volume_um3'] > brain_volume_lower_um3) & (props['axis_major_length_um'] > brain_axis_major_length_lower_um)
    brain_labels = props[brain_query]['label'].values
    brain_mask = np.isin(label_image, brain_labels)
    
    # 2. Create the directional brain outline
    brain_label_image = label(brain_mask)
    unique_brain_labels = np.unique(brain_label_image)
    unique_brain_labels = unique_brain_labels[unique_brain_labels != 0]

    print(f"Number of unique brain labels found: {len(unique_brain_labels)}") 
    if len(unique_brain_labels) >= 2:
        distances = []
        for lbl in tqdm(unique_brain_labels):
            dist = ndimage.distance_transform_edt(~(brain_label_image == lbl), sampling=spacing)
            distances.append(dist)
            
        dist_stack = np.stack(distances, axis=0)
        sorted_dists = np.partition(dist_stack, kth=1, axis=0)
        d1 = sorted_dists[0]
        d2 = sorted_dists[1]
        
        # Connect segments: within max_dilation_um of a segment, and within max_bridge_um of another
        directional_catchment = (d1 <= max_dilation_um) & (d2 <= max_bridge_um)
    else:
        directional_catchment = np.zeros_like(brain_mask, dtype=bool)
        
    # Construct the base outline of the brain (original brain + directional bridges)
    interim_brain_mask = brain_mask | directional_catchment
    
    # 3. Spatially-Adaptive Dilation of the brain outline
    # Find distance to brain outline and the coordinates of the closest edge voxel
    dist_interim, indices = ndimage.distance_transform_edt(
        ~interim_brain_mask, 
        sampling=spacing, 
        return_indices=True
    )
    
    # Pre-calculate coordinate grid distances to the center of the image (in physical micrometers)
    shape = mask_3d.shape
    c_z, c_y, c_x = shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0
    z_indices, y_indices, x_indices = np.indices(shape)
    
    dist_to_center = np.sqrt(
        ((z_indices - c_z) * spacing[0]) ** 2 +
        ((y_indices - c_y) * spacing[1]) ** 2 +
        ((x_indices - c_x) * spacing[2]) ** 2
    )
    
    # Lookup the physical distance from the image center to each voxel's closest brain edge
    edge_dist_to_center = dist_to_center[indices[0], indices[1], indices[2]]
    
    # Spatially adaptive threshold: higher dilation limit if voxel is further from center than its brain edge
    allowed_dilation = np.where(
        dist_to_center > edge_dist_to_center, 
        uniform_dilation_away_um, 
        uniform_dilation_towards_um
    )
    
    catchment_area = dist_interim <= allowed_dilation
    
    # 4. Find and Strictly Absorb Small Segments
    non_brain_labels = props[~brain_query]['label'].values
    
    # Find all labels that have at least one pixel OUTSIDE the catchment area.
    labels_outside = np.unique(label_image[~catchment_area])
    
    # A label is entirely inside if and only if none of its pixels are outside.
    to_absorb_labels = np.setdiff1d(non_brain_labels, labels_outside)
    
    # Create final masks
    to_absorb_mask = np.isin(label_image, to_absorb_labels)
    final_brain_mask = brain_mask | to_absorb_mask
    
    # The rest of the non-brain labels (not absorbed) are microglia candidates
    final_microglia_labels = np.setdiff1d(non_brain_labels, to_absorb_labels)
    final_microglia_candidates = np.isin(label_image, final_microglia_labels)
    
    # 5. Debug plotting
    if debug and plot_func is not None:
        plot_kwargs_to_use = plot_kwargs if plot_kwargs is not None else {}
        for zplane in final_microglia_candidates:
            plot_func(zplane[np.newaxis], **plot_kwargs_to_use)
        for zplane in final_brain_mask:
            plot_func(zplane[np.newaxis], **plot_kwargs_to_use)
            
    return final_brain_mask, final_microglia_candidates

def filter_microglia_by_morphology(
    microglia_candidates: np.ndarray,
    spacing: tuple,
    min_skeleton_length_um: float = 15.0,
    min_junctions: int = 2,
    min_endpoints: int = 3,
    min_svr: float = 0.05,
    microglia_volume_lower_um3: float = 50.0,    # 100 voxels * 0.5 um3/voxel = 50 um3
    microglia_volume_upper_um3: float = 2500.0,  # 5000 voxels * 0.5 um3/voxel = 2500 um3
    microglia_axis_major_length_upper_um: float = 25.0,     # 50 voxels * 0.5 um/voxel = 25 um
    debug: bool = False,
    plot_func: callable = None,
    plot_kwargs: dict = None
) -> tuple:
    """
    Filters potential microglia candidates using 3D skeletonization, 
    branching junctions, endpoints, Skeleton-to-Volume Ratio (SVR), 
    and physical size bounds to robustly handle oversaturated/overexposed images.
    
    Parameters
    ----------
    microglia_candidates : np.ndarray
        3D boolean or binary mask of microglia candidates (output from filter_microglia).
    spacing : tuple of floats
        Voxel spacing in micrometers in ZYX sequence: (dz, dy, dx).
    min_skeleton_length_um : float, optional
        Minimum physical skeleton length in micrometers. Default is 15.0 um.
    min_junctions : int, optional
        Minimum number of process branch points (nodes). Default is 2.
    min_endpoints : int, optional
        Minimum number of process tips (endpoints). Default is 3.
    min_svr : float, optional
        Minimum Skeleton-to-Volume Ratio (skeleton_voxels / area_voxels).
        Higher values represent branched/elongated structures; lower values represent solid blobs.
        Default is 0.05.
    microglia_volume_lower_um3 : float, optional
        Minimum volume of the object in cubic micrometers (um3). Default is 50.0 um3.
    microglia_volume_upper_um3 : float, optional
        Maximum volume of the object in cubic micrometers (um3). Default is 2500.0 um3.
    axis_major_length_upper_um : float, optional
        Maximum allowed physical major axis length in micrometers (um). Default is 25.0 um.
        
    Returns
    -------
    final_microglia_mask : np.ndarray
        3D boolean mask containing only the verified microglia segments.
    non_microglia_mask : np.ndarray
        3D boolean mask containing all rejected microglia candidates.
    microglia_props_df : pd.DataFrame
        DataFrame containing shape, skeleton, SVR, and branching features for each cell.
    """
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops_table
    from skimage.morphology import skeletonize
    import pandas as pd

    # 1. Label the candidate components directly
    label_image = label(microglia_candidates)
    candidate_labels = np.unique(label_image)
    candidate_labels = candidate_labels[candidate_labels != 0]
    
    if len(candidate_labels) == 0:
        empty_mask = np.zeros_like(microglia_candidates, dtype=bool)
        return empty_mask, empty_mask, pd.DataFrame()
    
    # 2. Perform 3D skeletonization on the candidates
    skeleton = skeletonize(microglia_candidates)
    
    # 3. Extract exact junctions and endpoints using 3D convolution neighborhood-count
    # A 3x3x3 kernel counts the active neighbors of each skeleton voxel
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0  # Exclude the center pixel itself
    
    # Convolve to get neighbor count for every skeleton voxel
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant', cval=0) * skeleton
    
    # Identify junctions (3+ neighbors) and endpoints (1 neighbor)
    endpoints_mask = (neighbor_count == 1) & skeleton
    junctions_mask = (neighbor_count >= 3) & skeleton
    
    # 4. Extract base region properties corrected for spacing (anisotropy)
    # By passing spacing=spacing:
    # - 'area' becomes the physical volume in cubic micrometers (um3)
    # - 'axis_major_length' becomes physical length in micrometers (um)
    props = pd.DataFrame(
        regionprops_table(label_image, properties=('label', 'area', 'axis_major_length'), spacing=spacing)
    )
    # Rename 'area' to 'volume_um3' for clarity
    props = props.rename(columns={'area': 'volume_um3', 'axis_major_length': 'axis_major_length_um'})
    
    # 5. Extract skeleton, endpoint, and junction counts using high-speed scipy.ndimage.sum
    skeleton_voxels = ndimage.sum(skeleton, label_image, index=candidate_labels)
    props['num_endpoints'] = ndimage.sum(endpoints_mask, label_image, index=candidate_labels)
    props['num_junctions'] = ndimage.sum(junctions_mask, label_image, index=candidate_labels)
    
    # Compute physical skeleton length in micrometers
    spacing_mean = np.mean(spacing)
    props['skeleton_length_um'] = skeleton_voxels * spacing_mean
    
    # Calculate Skeleton-to-Volume Ratio (SVR)
    # Using voxel count for SVR is mathematically equivalent and cleaner
    # Since 'volume_um3' = voxels * voxel_volume, we reconstruct voxels for the ratio
    voxel_volume = np.prod(spacing)
    area_voxels = props['volume_um3'] / voxel_volume
    props['svr'] = skeleton_voxels / area_voxels

    print(f"Volume: {props['volume_um3'].min()}, {props['volume_um3'].max()}")
    print(f"Skel Length: {props['axis_major_length_um'].min()}, {props['axis_major_length_um'].max()}")
    
    # 6. Apply multi-parametric physical size and branching filters
    microglia_filter = (
        (props['volume_um3'] >= microglia_volume_lower_um3) &
        (props['volume_um3'] <= microglia_volume_upper_um3) &
        (props['axis_major_length_um'] <= microglia_axis_major_length_upper_um) &
        (props['skeleton_length_um'] >= min_skeleton_length_um) &
        (props['num_junctions'] >= min_junctions) &
        (props['num_endpoints'] >= min_endpoints) &
        (props['svr'] >= min_svr)
    )
    
    verified_microglia_labels = props[microglia_filter]['label'].values
    
    # Generate final masks
    final_microglia_mask = np.isin(label_image, verified_microglia_labels)
    non_microglia_mask = microglia_candidates & ~final_microglia_mask
    
    # Add a column indicating if the candidate passed verification
    props['is_verified_microglia'] = microglia_filter
    
    # 7. Debug plotting
    if debug and plot_func is not None:
        plot_kwargs_to_use = plot_kwargs if plot_kwargs is not None else {}
        for zplane in final_microglia_mask:
            plot_func(zplane[np.newaxis], **plot_kwargs_to_use)
        for zplane in non_microglia_mask:
            plot_func(zplane[np.newaxis], **plot_kwargs_to_use)
            
    return final_microglia_mask, non_microglia_mask, props

def extract_microglia_from_zyx_stack(
    zyx_stack, 
    zyx_vox_spacing, 
    brain_volume_lower_um3=800,
    brain_axis_major_length_lower_um=80,
    max_dilation_um=20,
    max_bridge_um=20,
    uniform_dilation_towards_um=5,
    uniform_dilation_away_um=20,
    min_skeleton_length_um=0,
    min_junctions=2,
    min_endpoints=3,
    min_svr=0.2,
    microglia_volume_lower_um3=1000,
    microglia_volume_upper_um3=50000,
    microglia_axis_major_length_upper_um=200,
    debug=False,
    plot_func=None
    ):

    from microbiome.microglia import filter_microglia, filter_microglia_by_morphology
    # Dictionary of plotting options for make_projections
    plot_kwargs = {
        'im_mode': 'im_CM0_raw',
        'background': None,
        'vmin_mode': 'raw', 'vmin': 0,
        'vmax_mode': 'raw', 'vmax': 1,
        'cmap': 'gray'
    }

    if zyx_vox_spacing[0] == 5.0:
        brain_volume_lower_um3=800
        brain_axis_major_length_lower_um=80
        max_dilation_um=20
        max_bridge_um=20
        uniform_dilation_towards_um=5
        uniform_dilation_away_um=20
        min_skeleton_length_um=0
        min_junctions=0
        min_endpoints=0
        min_svr=0
        microglia_volume_lower_um3=1000
        microglia_volume_upper_um3=50000
        microglia_axis_major_length_upper_um=200

    if zyx_vox_spacing[0] == 1.0:
        brain_volume_lower_um3=800
        brain_axis_major_length_lower_um=80
        max_dilation_um=20
        max_bridge_um=20
        uniform_dilation_towards_um=5
        uniform_dilation_away_um=20
        min_skeleton_length_um=0
        min_junctions=1
        min_endpoints=2
        min_svr=0
        microglia_volume_lower_um3=1000
        microglia_volume_upper_um3=50000
        microglia_axis_major_length_upper_um=200

    brain_outline, microglia_candidates = filter_microglia(
        zyx_stack, 
        zyx_vox_spacing,
        brain_volume_lower_um3=brain_volume_lower_um3,
        brain_axis_major_length_lower_um=brain_axis_major_length_lower_um,
        max_dilation_um=max_dilation_um,
        max_bridge_um=max_bridge_um,
        uniform_dilation_towards_um=uniform_dilation_towards_um,
        uniform_dilation_away_um=uniform_dilation_away_um,
        debug=debug,
        plot_func=plot_func,
        plot_kwargs=plot_kwargs
        )

    microglia_mask, non_microglia_mask, _ = filter_microglia_by_morphology(
        microglia_candidates,
        spacing=zyx_vox_spacing,
        min_skeleton_length_um=min_skeleton_length_um,
        min_junctions=min_junctions,
        min_endpoints=min_endpoints,
        min_svr=min_svr,
        microglia_volume_lower_um3=microglia_volume_lower_um3,
        microglia_volume_upper_um3=microglia_volume_upper_um3,
        microglia_axis_major_length_upper_um=microglia_axis_major_length_upper_um,
        debug=debug,
        plot_func=plot_func,
        plot_kwargs=plot_kwargs
    )

    return brain_outline, microglia_candidates, microglia_mask, non_microglia_mask

def extract_and_store_microglia_from_zyx_stack(
    index, 
    tzyx_stack_zarr_path,
    vox_spacing,
    brain_outline_zarr_path,
    microglia_candidates_zarr_path,
    microglia_mask_zarr_path,
    non_microglia_mask_zarr_path
    ):

    import zarr
    from microbiome.microglia import extract_microglia_from_zyx_stack

    tzyx_stack_zarr = zarr.open(tzyx_stack_zarr_path)

    brain_outline, microglia_candidates, microglia_mask, non_microglia_mask =\
        extract_microglia_from_zyx_stack(
            tzyx_stack_zarr[index], 
            vox_spacing, 
            debug=False, 
            plot_func=None
            )

    brain_outline_zarr =  zarr.open(brain_outline_zarr_path, mode='r+') 
    microglia_candidates_zarr = zarr.open(microglia_mask_zarr_path, mode='r+')
    microglia_mask_zarr =  zarr.open(microglia_mask_zarr_path, mode='r+') 
    non_microglia_mask_zarr =  zarr.open(non_microglia_mask_zarr_path, mode='r+') 


    brain_outline_zarr[index] = brain_outline
    microglia_candidates_zarr[index] = microglia_candidates
    microglia_mask_zarr[index] = microglia_mask
    non_microglia_mask_zarr[index] = non_microglia_mask


def extract_and_store_microglia_from_tzyx_stack(    
    indices,
    tzyx_stack_zarr_path,
    vox_spacing,
    brain_outline_zarr_path,
    microglia_candidates_zarr_path,
    microglia_mask_zarr_path,
    non_microglia_mask_zarr_path
):

    import zarr

    if isinstance(indices, int):
        indices = [indices]

    tzyx_stack_zarr = zarr.open(tzyx_stack_zarr_path)

    for zarr_path in [brain_outline_zarr_path, microglia_candidates_zarr_path, microglia_mask_zarr_path, non_microglia_mask_zarr_path]:

        if not os.path.exists(zarr_path):
            zarr.create(
                shape=tzyx_stack_zarr.shape, 
                chunks=tzyx_stack_zarr.chunks, 
                dtype=tzyx_stack_zarr.dtype, 
                store=zarr_path
            )

    return [extract_and_store_microglia_from_zyx_stack(
        index, 
        tzyx_stack_zarr_path,
        vox_spacing,
        brain_outline_zarr_path,
        microglia_candidates_zarr_path,
        microglia_mask_zarr_path,
        non_microglia_mask_zarr_path
        ) for index in indices]

def extract_microglia_candidates_using_simple_outline_computation(
    spim_dset, 
    recompute=False,
    recompute_microglia_labels=False,
    overwrite_microglia_characterization=False
    ):

    from microbiome.microglia import extract_or_load_microglia_labels, extract_or_load_microglia_characterization, generate_or_load_microglia_stack, generate_or_load_outline_stack
    
    if recompute is not None:
        recompute_microglia_labels = recompute
        overwrite_microglia_characterization = recompute

    extract_or_load_microglia_labels(spim_dset, recompute=recompute_microglia_labels)
    extract_or_load_microglia_characterization(spim_dset, overwrite=overwrite_microglia_characterization)  ## 30 seconds per dataset

    spim_dset.small_objs_df['time_h'] = spim_dset.small_objs_df['time_s'] / 3600

    ## Generate an outline stack to register with
    ### (1) Find microglia, then subtract it from the data
    ### (2) Find the outline stack directly

    spim_dset.microglia_stack = generate_or_load_microglia_stack(
        spim_dset, 
        recompute=False, 
        copy_data_to_local=True,
        brain_area_lower=spim_dset.brain_area_lower, 
        brain_axis_major_length_lower=brain_axis_major_length_lower, 
        microglia_area_lower=spim_dset.microglia_area_lower,
        microglia_area_upper=spim_dset.microglia_area_upper, 
        microglia_axis_major_length_upper=microglia_axis_major_length_upper
        )
    
    generate_or_load_outline_stack(
        spim_dset, 
        recompute=False,
        copy_data_to_local=True
    )

def extract_microglia_candidates_using_complex_outline_computation(
    spim_dset, 
    recompute=False,
    copy_data_to_local=True,
    parallelize=True
):

    from microbiome.microglia import extract_and_store_microglia_from_tzyx_stack
    from microbiome._vendor.analysis_toolbox.parallel import dask_client_compute

    from functools import partial
    import shutil

    spim_dset.im_CM0_raw_rechunked_path = f'{spim_dset.procpath}im_CM0_raw_rechunked.zarr'
    spim_dset.brain_outline_zarr_path = f'{spim_dset.procpath}brain_outline.zarr'
    spim_dset.microglia_candidates_zarr_path = f'{spim_dset.procpath}microglia_candidates.zarr'
    spim_dset.microglia_mask_zarr_path = f'{spim_dset.procpath}microglia_mask.zarr'
    spim_dset.non_microglia_mask_zarr_path = f'{spim_dset.procpath}non_microglia_mask.zarr'

    if recompute:

        for output_path in [
            spim_dset.brain_outline_zarr_path,
            spim_dset.microglia_candidates_zarr_path,
            spim_dset.microglia_mask_zarr_path,
            spim_dset.non_microglia_mask_zarr_path]:

            if os.path.exists(output_path):
                shutil.rmtree(output_path)


    if not os.path.exists(spim_dset.microglia_mask_zarr_path):

        if not os.path.exists(spim_dset.im_CM0_raw_rechunked_path):
            
            from microbiome.microglia import rechunk_nikon_twophoton_data

            rechunk_nikon_twophoton_data(
                spim_dset.im_CM0_raw.data, 
                spim_dset.im_CM0_raw.path, 
                spim_dset.im_CM0_raw_rechunked_path, 
                copy_data_to_local=copy_data_to_local
                )

        if copy_data_to_local:

            from microbiome._vendor.analysis_toolbox.fileio import make_local_copy

            data_to_process = make_local_copy(spim_dset.im_CM0_raw_rechunked_path, TEMP_DIR)

        else:
            data_to_process = spim_dset.im_CM0_raw_rechunked_path


        extract_and_store_microglia_from_tzyx = partial(
            extract_and_store_microglia_from_tzyx_stack,
            tzyx_stack_zarr_path = data_to_process,
            vox_spacing=spim_dset.im_CM0_raw.vox_spacing,
            brain_outline_zarr_path=spim_dset.brain_outline_zarr_path,
            microglia_candidates_zarr_path=spim_dset.microglia_candidates_zarr_path,
            microglia_mask_zarr_path=spim_dset.microglia_mask_zarr_path,
            non_microglia_mask_zarr_path=spim_dset.non_microglia_mask_zarr_path

        )

        from microbiome._vendor.analysis_toolbox.parallel import dask_client_compute

        if parallelize:

            if spim_dset.im_CM0_raw.vox_spacing[0] == 1:
                num_workers = 1
            elif spim_dset.im_CM0_raw.vox_spacing[0] == 5:
                num_workers = 24

            result, done = dask_client_compute(
                list(range(len(spim_dset.im_CM0_raw.data))),
                func=extract_and_store_microglia_from_tzyx,
                return_done=True,
                mappable_id="index",
                num_workers=num_workers,
                num_cores_per_worker=1,
            )

        else:
            extract_and_store_microglia_from_tzyx(list(range(len(spim_dset.im_CM0_raw.data))))

def extract_and_characterize_objects(
    mask_3d, 
    vox_spacing, 
    properties=('label', 'area', 'axis_major_length')
    ):

    from skimage.measure import label, regionprops_table

    # 1. Separate objects by physical size and shape properties using spacing=spacing
    labels = label(mask_3d)
    props = pd.DataFrame(
        regionprops_table(labels, properties=properties, spacing=vox_spacing)
    )
    # Rename columns to reflect physical units
    props = props.rename(columns={'area': 'volume_um3', 'axis_major_length': 'axis_major_length_um'})

    return labels, props

def extract_and_characterize_microglia(
    spim_dset, 
    microglia_mask, 
    baseline_microglia_mask=None,
    labels_savepath='',
    microglia_df_savepath=''
    ):

    import skimage as ski
    from microbiome._vendor.analysis_toolbox.fileio import fill_zarr_along_axis0

    small_objs_list = []

    if not labels_savepath:
        labels_list = []

    if baseline_microglia_mask is not None:
        n_timepoints = len(spim_dset.im_CM0_raw.t_axis_s) + 1
    else:
        n_timepoints = len(spim_dset.im_CM0_raw.t_axis_s)

    nrows_counter = 0
    nstacks_counter = 0

    # - area, solidity and extent are the holy trinity
    # - however, my images are overexposed, so it fills in the gaps between microglia branches, 
    #   making them look like solid blobs
    # - I think feret_diameter_max could be useful
    properties = [
        'label', 
        'area', 
        'centroid', 
        'coords', 
        'image', 
        'axis_major_length',
        # 'feret_diameter_max', ## if your objects are large (like a brain outline), the number of points involved in that calculation can explode
        # 'solidity',  # Solidity: how "holey" or "branched" the object is
        # 'extent',  # Extent: how much of the bounding box is filled by the object
        ]

    if baseline_microglia_mask is not None:

        labels, small_objs_df = extract_and_characterize_objects(
            baseline_microglia_mask, 
            spim_dset.im_CM0_raw.vox_spacing, 
            properties=properties
            )

        format_regionprops_table(
            small_objs_df, 
            timepoint = -15 * 60,   ## 15 mins before
            expt_id=f"{spim_dset.expt_date.strftime('%Y%m%d')} {spim_dset.fish_num}"
            )  

        nrows_timepoint = len(small_objs_df)
        nrows_counter += nrows_timepoint 

        if labels_savepath: 
            fill_zarr_along_axis0(
                labels_savepath, 
                (n_timepoints, ) + labels.shape, 
                nstacks_counter, 
                labels.astype(np.int16), 
                dtype=np.int16, 
                chunks=(1, ) + labels.shape)

            nstacks_counter += 1
            
        else:
            labels_list.append(labels)

        small_objs_list.append(small_objs_df)

    for t, mask in tqdm(zip(spim_dset.im_CM0_raw.t_axis_s, microglia_mask), total=len(spim_dset.im_CM0_raw.t_axis_s)):

        print(f"Processing time point {t} seconds / {t/3600} hours...")

        labels, small_objs_df = extract_and_characterize_objects(
            mask, 
            spim_dset.im_CM0_raw.vox_spacing, 
            properties=properties
            )

        format_regionprops_table(
            small_objs_df, 
            timepoint = t,   ## current time point
            expt_id=f"{spim_dset.expt_date.strftime('%Y%m%d')} {spim_dset.fish_num}"
            )  

        nrows_timepoint = len(small_objs_df)
        print(nrows_timepoint)
        nrows_counter += nrows_timepoint 
        
        if labels_savepath: 
            fill_zarr_along_axis0(
                labels_savepath, 
                (n_timepoints, ) + labels.shape, 
                nstacks_counter, 
                labels.astype(np.int16), 
                dtype=np.int16, 
                chunks=(1, ) + labels.shape)

            nstacks_counter += 1

        else:
            labels_list.append(labels)

        small_objs_list.append(small_objs_df)

    if not labels_savepath:

        labels_stack = np.stack(labels_list, axis=0)

    small_objs_df = pd.concat(small_objs_list, ignore_index=True)
    assert nrows_counter == len(small_objs_df)

    if microglia_df_savepath:
        small_objs_df.to_pickle(microglia_df_savepath)

    return small_objs_df

def _extract_period_info(metadata_or_periods):
    """
    Accept either the full ND2 metadata object (which has a ``periods`` attribute)
    or a plain list/tuple of Period objects.  Returns a list of
    ``(interval_h, count, duration_h)`` for each period.
    """
    import numpy as np

    if hasattr(metadata_or_periods, "periods"):
        periods = metadata_or_periods.periods
    else:
        periods = metadata_or_periods

    if periods is None or len(periods) == 0:
        raise ValueError("No periods found in the supplied metadata.")

    info = []
    for p in periods:
        interval_h = float(p.periodDiff.avg) / 3_600_000.0   # ms -> hours
        count = int(getattr(p, "count", 0))
        duration_h = float(getattr(p, "durationMs", 0)) / 3_600_000.0
        if count <= 0:
            raise ValueError("Each Period must have a positive `count`.")
        if interval_h <= 0:
            raise ValueError("Each Period must have a positive average spacing.")
        info.append((interval_h, count, duration_h))
    return info


def _build_full_series_from_metadata(metadata_or_periods):
    """
    Construct a *full* timestamp series from the ND2 metadata.
    The short epoch is generated until the first value that exceeds the
    duration of the first period, then the long epoch starts from the
    epoch duration (not from the overshoot).  This produces a series like
    ``..., 0.945, 1.008, 1.5, 2.0, ...``.
    """
    import numpy as np

    period_info = _extract_period_info(metadata_or_periods)
    if len(period_info) < 2:
        raise ValueError("Metadata must contain at least two periods (short & long).")

    short_int, _, short_dur = period_info[0]
    long_int, _, _ = period_info[1]

    # Short epoch – include the first overshoot.
    short_block = np.arange(0,
                            short_dur + short_int + 1e-12,
                            short_int,
                            dtype=float)
    if short_block[-1] <= short_dur:
        short_block = np.append(short_block, short_block[-1] + short_int)

    # Long epoch – starts from epoch duration + long interval.
    long_start = short_dur + long_int
    long_block = np.arange(long_start,
                           long_start + long_int * 1000,
                           long_int,
                           dtype=float)

    return np.concatenate([short_block, long_block])


def _is_expected_timestamp_array(arr, tol_short=0.05, tol_long=0.10):
    """
    Quick sanity-check for the two-segment pattern.
    """
    import numpy as np

    a = np.asarray(arr, dtype=float)
    if a.ndim != 1 or a.size < 2:
        return False

    split_idx = np.searchsorted(a, 1.0, side='right')
    if split_idx == 0 or split_idx == a.size:
        return False

    short_diffs = np.diff(a[:split_idx])
    if short_diffs.size == 0 or np.any(short_diffs == 0) or not np.allclose(short_diffs,
                                              short_diffs[0],
                                              rtol=tol_short):
        return False

    long_diffs = np.diff(a[split_idx:])
    if long_diffs.size == 0 or np.any(long_diffs == 0) or not np.allclose(long_diffs,
                                              long_diffs[0],
                                              rtol=tol_long):
        return False

    return True


def _manual_timestamps_from_durations(intervals_h,
                                     durations_h,
                                     target_len):
    """
    Build timestamps from user‑supplied intervals and the duration of the short
    epoch.  The short epoch is generated until the first overshoot, then the
    long epoch starts from the epoch duration.
    """
    import numpy as np

    if len(intervals_h) < 2:
        raise ValueError("At least two intervals (short + long) must be provided.")
    if len(durations_h) == 0:
        raise ValueError("The duration of the short segment must be provided.")
    if target_len <= 0:
        raise ValueError("target_len must be a positive integer.")

    short_int, long_int = intervals_h[0], intervals_h[1]
    epoch_dur = durations_h[0]

    # Short epoch – include the overshoot.
    short_block = np.arange(0,
                            epoch_dur + short_int + 1e-12,
                            short_int,
                            dtype=float)
    if short_block[-1] <= epoch_dur:
        short_block = np.append(short_block, short_block[-1] + short_int)

    if short_block.size >= target_len:
        return short_block[:target_len]

    # Long epoch – starts from epoch duration + long interval.
    remaining = target_len - short_block.size
    long_start = epoch_dur + long_int
    long_block = np.arange(long_start,
                          long_start + long_int * remaining,
                          long_int,
                          dtype=float)
    if len(long_block) < remaining:
        long_block = np.append(long_block, long_block[-1] + long_int)

    result = np.concatenate([short_block, long_block])

    # Enforce exact length.
    if result.shape[0] > target_len:
        result = result[:target_len]
    elif result.shape[0] < target_len:
        extra = np.arange(result[-1] + long_int,
                          result[-1] + long_int *
                          (target_len - result.shape[0] + 1),
                          long_int,
                          dtype=float)[:target_len - result.shape[0]]
        result = np.concatenate([result, extra])

    return result


def correct_timestamps(original,
                      metadata_or_periods,
                      manual_intervals_h=None,
                      manual_durations_h=None):
    """
    Three-step pipeline that also **prints which step actually fixed the array**.

    1) If the original array already matches the expected pattern, it is
       returned unchanged (step 1).
    2) Otherwise try to rebuild the full series from ND2 metadata and replace
       only the tail that is wrong (step 2).
    3) If step 2 still does not match, fall back to the manual intervals/
       durations (step 3).

    Parameters
    ----------
    original : Sequence[float]
        The timestamp array you currently have.
    metadata_or_periods : ND2 metadata object **or** list of Period objects.
    manual_intervals_h : list[float] | None
        ``[short_interval, long_interval]`` in hours.
    manual_durations_h : list[float] | None
        ``[duration_of_short_segment, …]`` in hours.

    Returns
    -------
    np.ndarray
        A corrected timestamp array (hours) that passes the sanity check.
    """
    import numpy as np

    orig = np.asarray(original, dtype=float)

    # -------------------------------------------------------------
    # Step 1 – original check
    # ------------------------------------------datasets.microglia_props_df.loc[:, 'volume_um3']-------------------
    if _is_expected_timestamp_array(orig):
        print("Step 1: original array is already valid.")
        return orig.copy()

    # -------------------------------------------------------------
    # Step 2 completed; proceed to step 3
    # -------------------------------------------------------------
    try:
        # Extract interval info to decide if metadata is usable
        period_info = _extract_period_info(metadata_or_periods)
        short_int = period_info[0][0]
        # If the short interval is zero or negative, cannot use metadata series
        if short_int <= 0:
            raise ValueError("Zero or negative short interval, skipping metadata reconstruction.")
        full_meta = _build_full_series_from_metadata(metadata_or_periods)
        # Proceed with tail replacement as before
        corrected = orig.copy()
        if len(corrected) >= 2:
            new_vals = full_meta[-2:]
            if np.abs(corrected[-2] - new_vals[0]) > 1e-6:
                corrected[-2] = new_vals[0]
            if np.abs(corrected[-1] - new_vals[1]) > 1e-6:
                corrected[-1] = new_vals[1]
        else:
            corrected = full_meta[:len(orig)].copy()
        if np.allclose(orig[:-2], corrected[:-2], atol=1e-6) and _is_expected_timestamp_array(corrected):
            print("Step 2: fixed using ND2 metadata (tail replacement).")
            return corrected
    except Exception:
        pass  # metadata step failed or not applicable

    # -------------------------------------------------------------
    # Step 3 – manual fallback (full reconstruction for all‑zero arrays)
    # -------------------------------------------------------------
    if manual_intervals_h is not None and manual_durations_h is not None:
        target_len = len(orig)
        manual_series = _manual_timestamps_from_durations(
            manual_intervals_h,
            manual_durations_h,
            target_len,
        )
        # If the original array is all zeros (or contains no valid pattern), replace entirely
        if np.allclose(orig, 0, atol=1e-9):
            if _is_expected_timestamp_array(manual_series):
                print("Step 3: fixed using manual intervals/durations (full reconstruction for zero array).")
                return manual_series
            else:
                raise RuntimeError("Manual construction failed for zero array.")
        # Otherwise, attempt tail replacement as before
        corrected = orig.copy()
        corrected[-2:] = manual_series[-2:]
        if _is_expected_timestamp_array(corrected):
            print("Step 3: fixed using manual intervals/durations (tail replacement).")
            return corrected
        # If tail replacement fails, fallback to full reconstruction
        if _is_expected_timestamp_array(manual_series):
            print("Step 3: fixed using manual intervals/durations (full reconstruction fallback).")
            return manual_series
        raise RuntimeError("Manual construction still does not pass the sanity check.")
    else:
        raise RuntimeError(
            "Could not generate a valid timestamp array. No manual intervals/durations supplied."
        )

def analyze_microglia_mixed_model(df_wide, time_column='time_hr', model_type='interactive', reference_condition="Sparos"):
    """
    Reshapes a wide dataframe into long format and runs a Linear Mixed-Effects Model.
    
    Parameters:
    - df_wide: The wide pandas DataFrame with MultiIndex ['condition', 'expt']
    - model_type: 'interactive' (tests for different slopes) or 'additive' (tests for parallel vertical shifts)
    - reference_condition: The condition to use as the control/baseline.
    
    Returns:
    - df_long: The cleaned, reshaped long dataframe used for the model.
    - results: The fitted statsmodels results object.
    """
    import pandas as pd
    import statsmodels.formula.api as smf
    
    # 1. Convert wide to long
    df_long = df_wide.stack().reset_index(name='count')
    
    # 2. BULLETPROOFING: Aggressively drop all NaNs and reset the internal row index 
    # to guarantee statsmodels doesn't trip over missing data length mismatches
    df_long = df_long.dropna().reset_index(drop=True)
    
    # 3. Determine the formula based on the kwarg
    if model_type == 'interactive':
        formula = f"count ~ C(condition, Treatment('{reference_condition}')) * {time_column}"
        print(f"Running Interactive Model (Testing for Different Slopes) - Baseline: {reference_condition}")
    
    elif model_type == 'additive':
        formula = f"count ~ C(condition, Treatment('{reference_condition}')) + {time_column}"
        print(f"Running Additive Model (Testing for Parallel Vertical Shifts) - Baseline: {reference_condition}")
    
    else:
        raise ValueError("model_type must be either 'interactive' or 'additive'")
        
    # 4. Run the model
    model = smf.mixedlm(formula, data=df_long, groups="expt")
    results = model.fit()
    
    print(results.summary())
    return df_long, results

def add_mixed_model_bracket_to_ax(ax, df_wide, results, group1, group2=None, reference_condition="Sparos", model_type=None, orientation="horizontal"):
    """
    Annotates a single matplotlib Axes object with significance brackets
    based on pre-calculated linear mixed-effects model results.
    
    Parameters:
    - ax: Matplotlib Axes object.
    - df_wide: Wide-format pandas DataFrame with MultiIndex ['condition', 'expt'] and time columns.
    - results: Statsmodels results object or a dictionary of p-values.
    - group1: Group name(s) (list of conditions for combined plot, or single condition string for pairwise plot).
    - group2: Second group name (string, if pairwise plot).
    - reference_condition: Baseline condition for comparisons.
    - model_type: 'interactive', 'additive', or None (auto-detected).
    - orientation: 'horizontal' (top of plot) or 'vertical' (right side of plot).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    from itertools import combinations
    
    def convert_pvalue_to_asterisks(pvalue, hide_ns=False):
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        if hide_ns:
            return ""
        else:
            return "ns"
    
    # Helper to extract or compute pairwise p-value from model results
    def get_pval(cond1, cond2):
        if results is None:
            return None
            
        # 1. If results is a dict (user provided direct p-values)
        if isinstance(results, dict):
            for key, val in results.items():
                if isinstance(key, tuple):
                    if (cond1 in key and cond2 in key):
                        return val
                elif isinstance(key, str):
                    if (key == cond1 or key == cond2):
                        return val
            return None
            
        # 2. If results is a statsmodels results object (compute contrast using params and cov)
        if hasattr(results, 'pvalues'):
            # Detect model type if not explicitly provided
            m_type = model_type
            if m_type is None:
                m_type = 'additive'
                for term in results.params.index:
                    if 'time_hr' in term and ':' in term:
                        m_type = 'interactive'
                        break
            
            def find_term(cond):
                # Search for the term containing the condition name wrapped in treatment format [T.cond]
                # to avoid false positives like matching "Sparos" inside "Treatment('Sparos')"
                target = f"[T.{cond}]"
                for term in results.params.index:
                    if target in term:
                        if (m_type == 'interactive' and 'time_hr' in term) or (m_type == 'additive' and 'time_hr' not in term and 'Intercept' not in term):
                            return term
                return None
                
            term1 = find_term(cond1)
            term2 = find_term(cond2)
            
            if term1 is None and term2 is None:
                return 1.0
            elif term1 is None:
                # cond1 is the reference condition
                if term2 in results.pvalues:
                    return float(results.pvalues[term2])
            elif term2 is None:
                # cond2 is the reference condition
                if term1 in results.pvalues:
                    return float(results.pvalues[term1])
            else:
                # Neither is reference, compute difference z-test using params and cov_params
                try:
                    beta1 = results.params[term1]
                    beta2 = results.params[term2]
                    diff = beta1 - beta2
                    
                    cov = results.cov_params()
                    var1 = cov.loc[term1, term1]
                    var2 = cov.loc[term2, term2]
                    cov12 = cov.loc[term1, term2]
                    
                    var_diff = var1 + var2 - 2 * cov12
                    if var_diff > 0:
                        z = diff / np.sqrt(var_diff)
                        return float(2 * (1 - stats.norm.cdf(abs(z))))
                except Exception as e:
                    print(f"Error calculating pairwise contrast for {cond1} vs {cond2}: {e}")
                    
        return None

    # Helper to find condition color from plotted lines or legend
    def get_condition_color(ax, cond):
        # 1. Try to find the color from the legend handles and labels
        legend = ax.get_legend()
        if legend is not None:
            labels = [t.get_text() for t in legend.get_texts()]
            handles = getattr(legend, 'legend_handles', None)
            if handles is None:
                handles = getattr(legend, 'legendHandles', None)
            if handles is not None:
                for handle, label in zip(handles, labels):
                    if cond == label or label.startswith(cond + " ") or label.startswith(cond + " ("):
                        if hasattr(handle, 'get_color'):
                            return handle.get_color()
                        if hasattr(handle, 'get_edgecolor'):
                            ec = handle.get_edgecolor()
                            if len(ec) > 0:
                                return ec
                        if hasattr(handle, 'lines'):
                            line = handle.lines[0]
                            if hasattr(line, 'get_color'):
                                return line.get_color()
                                
        # 2. Try ax.get_legend_handles_labels()
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if cond == label or label.startswith(cond + " ") or label.startswith(cond + " ("):
                if hasattr(handle, 'get_color'):
                    return handle.get_color()
                if hasattr(handle, 'get_edgecolor'):
                    ec = handle.get_edgecolor()
                    if len(ec) > 0:
                        return ec
                if hasattr(handle, 'lines'):
                    line = handle.lines[0]
                    if hasattr(line, 'get_color'):
                        return line.get_color()
                        
        # 3. Fallback to searching lines in ax (in case they are not in legend)
        for line in ax.get_lines():
            label = line.get_label()
            if label == cond or label.startswith(cond + " ") or label.startswith(cond + " ("):
                return line.get_color()
                
        # 4. Fallback to matching a line whose label contains cond
        for line in ax.get_lines():
            label = line.get_label()
            if cond in label:
                return line.get_color()
                
        return "black" # absolute fallback

    # Helper to reposition legend to lower left outside the plot
    def reposition_legend(ax, offset_x=1.02):
        legend = ax.get_legend()
        if legend is not None:
            labels = [t.get_text() for t in legend.get_texts()]
            handles = getattr(legend, 'legend_handles', None)
            if handles is None:
                handles = getattr(legend, 'legendHandles', None)
            if handles is None:
                handles, _ = ax.get_legend_handles_labels()
                
            legend.remove()
            
            # Recreate legend outside on the right, hanging below the bottom edge of the axes
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(offset_x, 0.0), frameon=True, facecolor='white', edgecolor='none', framealpha=0.8)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # Find first and last timepoints within the current x-limits
    try:
        valid_cols = [c for c in df_wide.columns if float(c) <= xlim[1] and float(c) >= xlim[0]]
        if len(valid_cols) > 0:
            t_first = float(valid_cols[0])
            t_last = float(valid_cols[-1])
            t_col_last = valid_cols[-1]
        else:
            t_first = float(df_wide.columns[0])
            t_last = float(df_wide.columns[-1])
            t_col_last = df_wide.columns[-1]
    except (ValueError, TypeError, IndexError):
        t_first = xlim[0] + 0.1 * x_range
        t_last = xlim[1] - 0.1 * x_range
        t_col_last = df_wide.columns[-1]

    legend_offset_x = 1.02

    # --- Case A: Combined Plot (group2 is None) ---
    if group2 is None:
        conditions = list(group1) if isinstance(group1, (list, tuple, np.ndarray, pd.Index)) else [group1]
        other_conditions = [c for c in conditions if c != reference_condition]
        
        if orientation == "horizontal":
            for j, cond in enumerate(other_conditions):
                pval = get_pval(reference_condition, cond)
                if pval is not None:
                    sig_label = convert_pvalue_to_asterisks(pval, hide_ns=False)
                    
                    color1 = get_condition_color(ax, reference_condition)
                    color2 = get_condition_color(ax, cond)
                    t_mid = (t_first + t_last) / 2.0
                    
                    y_b = ylim[1] + (0.02 + 0.06 * j) * y_range
                    tick_height = 0.015 * y_range
                    
                    # Draw left half of the bracket in color1 (clip_on=False)
                    ax.plot([t_first, t_mid], [y_b, y_b], color=color1, lw=1.2, clip_on=False)
                    ax.plot([t_first, t_first], [y_b, y_b - tick_height], color=color1, lw=1.2, clip_on=False)
                    
                    # Draw right half of the bracket in color2 (clip_on=False)
                    ax.plot([t_mid, t_last], [y_b, y_b], color=color2, lw=1.2, clip_on=False)
                    ax.plot([t_last, t_last], [y_b, y_b - tick_height], color=color2, lw=1.2, clip_on=False)
                    
                    # Add text in neutral color (clip_on=False)
                    x_mid = (t_first + t_last) / 2.0
                    ax.text(x_mid, y_b + 0.005 * y_range, sig_label, ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', clip_on=False)
                    
        elif orientation == "vertical":
            # Offset legend dynamically to the right of the vertical brackets
            legend_offset_x = 1.02 + 0.10 * len(other_conditions)
            
            for j, cond in enumerate(other_conditions):
                pval = get_pval(reference_condition, cond)
                if pval is not None:
                    sig_label = convert_pvalue_to_asterisks(pval, hide_ns=False)
                    
                    color1 = get_condition_color(ax, reference_condition)
                    color2 = get_condition_color(ax, cond)
                    
                    # Calculate y endpoints at t_col_last
                    try:
                        y1 = df_wide.loc[reference_condition, t_col_last].mean()
                        y2 = df_wide.loc[cond, t_col_last].mean()
                        if pd.isna(y1) or pd.isna(y2):
                            raise ValueError
                    except Exception:
                        y1 = np.nanmean(df_wide.loc[reference_condition].values)
                        y2 = np.nanmean(df_wide.loc[cond].values)
                        
                    y_min = min(y1, y2)
                    y_max = max(y1, y2)
                    
                    x_b = xlim[1] + (0.02 + 0.09 * j) * x_range
                    tick_width = 0.015 * x_range
                    x_text = x_b + 0.01 * x_range
                    
                    if y1 > y2:
                        color_upper = color1
                        color_lower = color2
                    else:
                        color_upper = color2
                        color_lower = color1
                        
                    y_mid = (y_min + y_max) / 2.0
                    
                    # Draw upper half of bracket (clip_on=False)
                    ax.plot([x_b, x_b], [y_mid, y_max], color=color_upper, lw=1.2, clip_on=False)
                    ax.plot([x_b - tick_width, x_b], [y_max, y_max], color=color_upper, lw=1.2, clip_on=False)
                    
                    # Draw lower half of bracket (clip_on=False)
                    ax.plot([x_b, x_b], [y_min, y_mid], color=color_lower, lw=1.2, clip_on=False)
                    ax.plot([x_b - tick_width, x_b], [y_min, y_min], color=color_lower, lw=1.2, clip_on=False)
                    
                    # Add text in neutral color (clip_on=False)
                    ax.text(x_text, y_mid, sig_label, ha='left', va='center', fontsize=9, fontweight='bold', color='black', clip_on=False)

    # --- Case B: Pairwise Plot (group2 is not None) ---
    else:
        cond1 = group1
        cond2 = group2
        pval = get_pval(cond1, cond2)
        if pval is not None:
            sig_label = convert_pvalue_to_asterisks(pval, hide_ns=False)
            color1 = get_condition_color(ax, cond1)
            color2 = get_condition_color(ax, cond2)
            
            if orientation == "horizontal":
                y_b = ylim[1] + 0.03 * y_range
                tick_height = 0.02 * y_range
                t_mid = (t_first + t_last) / 2.0
                
                # Draw left half of bracket (clip_on=False)
                ax.plot([t_first, t_mid], [y_b, y_b], color=color1, lw=1.2, clip_on=False)
                ax.plot([t_first, t_first], [y_b, y_b - tick_height], color=color1, lw=1.2, clip_on=False)
                
                # Draw right half of bracket (clip_on=False)
                ax.plot([t_mid, t_last], [y_b, y_b], color=color2, lw=1.2, clip_on=False)
                ax.plot([t_last, t_last], [y_b, y_b - tick_height], color=color2, lw=1.2, clip_on=False)
                
                # Add text (clip_on=False)
                x_mid = (t_first + t_last) / 2.0
                ax.text(x_mid, y_b + 0.005 * y_range, sig_label, ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', clip_on=False)
                
            elif orientation == "vertical":
                # Offset legend to the right of the vertical bracket
                legend_offset_x = 1.12
                
                x_b = xlim[1] + 0.03 * x_range
                tick_width = 0.02 * x_range
                x_text = x_b + 0.01 * x_range
                
                # Calculate endpoints at t_col_last
                try:
                    y1 = df_wide.loc[cond1, t_col_last].mean()
                    y2 = df_wide.loc[cond2, t_col_last].mean()
                    if pd.isna(y1) or pd.isna(y2):
                        raise ValueError
                except Exception:
                    y1 = np.nanmean(df_wide.loc[cond1].values)
                    y2 = np.nanmean(df_wide.loc[cond2].values)
                    
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                
                if y1 > y2:
                    color_upper = color1
                    color_lower = color2
                else:
                    color_upper = color2
                    color_lower = color1
                    
                y_mid = (y_min + y_max) / 2.0
                
                # Draw upper half of bracket (clip_on=False)
                ax.plot([x_b, x_b], [y_mid, y_max], color=color_upper, lw=1.2, clip_on=False)
                ax.plot([x_b - tick_width, x_b], [y_max, y_max], color=color_upper, lw=1.2, clip_on=False)
                
                # Draw lower half of bracket (clip_on=False)
                ax.plot([x_b, x_b], [y_min, y_mid], color=color_lower, lw=1.2, clip_on=False)
                ax.plot([x_b - tick_width, x_b], [y_min, y_min], color=color_lower, lw=1.2, clip_on=False)
                
                # Add text (clip_on=False)
                ax.text(x_text, y_mid, sig_label, ha='left', va='center', fontsize=9, fontweight='bold', color='black', clip_on=False)

    reposition_legend(ax, offset_x=legend_offset_x)


def add_mixed_model_brackets(figs, axs, df_wide, results, reference_condition="Sparos", model_type=None, orientation="horizontal"):
    """
    Annotates the figures and axes returned by make_plots() with significance brackets
    based on pre-calculated linear mixed-effects model results.
    
    Parameters:
    - figs: List of matplotlib Figure objects returned by make_plots().
    - axs: List of matplotlib Axes objects returned by make_plots().
    - df_wide: Wide-format pandas DataFrame with MultiIndex ['condition', 'expt'] and time columns.
    - results: Statsmodels results object or a dictionary of p-values.
    - reference_condition: Baseline condition for comparisons.
    - model_type: 'interactive', 'additive', or None (auto-detected).
    - orientation: 'horizontal' (top of plot) or 'vertical' (right side of plot).
    """
    import numpy as np
    from itertools import combinations
    
    conditions = df_wide.index.get_level_values(0).unique().tolist()
    condition_pairs = list(combinations(conditions, 2))
    
    # 1. Combined plot (axs[0])
    if len(axs) > 0:
        ax = axs[0]
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()[0]
        add_mixed_model_bracket_to_ax(ax, df_wide, results, group1=conditions, group2=None, reference_condition=reference_condition, model_type=model_type, orientation=orientation)
        
    # 2. Pairwise plots (axs[1:])
    for i, (cond1, cond2) in enumerate(condition_pairs):
        idx_ax = i + 1
        if idx_ax >= len(axs):
            break
        ax = axs[idx_ax]
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()[0]
        add_mixed_model_bracket_to_ax(ax, df_wide, results, group1=cond1, group2=cond2, reference_condition=reference_condition, model_type=model_type, orientation=orientation)


def annotate_model_results(ax, data, x, y, order, results, reference_condition="Sparos", model_type="additive", continuous_var="time_hr", text_format="star", **kwargs):
    """
    Wrapper around annotate_custom_pvalues that automatically extracts the correct 
    p-values from statsmodels results (AUC, Additive LMM, or Interactive LMM).
    
    Parameters:
    - ax, data, x, y, order: Standard seaborn plotting arguments.
    - results: The fitted statsmodels results object.
    - reference_condition: The control group to compare against.
    - model_type: 'auc', 'additive', or 'interactive'.
    - continuous_var: The name of the time variable (only used if model_type is 'interactive').
    - text_format: Format for the p-value annotations. Supported options:
        - "star" (default): significance brackets with asterisks.
        - "simple": simplified p-values.
        - "full": full p-values (e.g. p = 0.05).
    - **kwargs: Extra formatting parameters passed to statannotations.Annotator.configure
      (e.g., pvalue_format_string, simple_format_string, pvalue_thresholds, p_capitalized, p_separators).
    """
    
    # 1. Automatically generate the box pairs comparing everything to the reference
    box_pairs = [(reference_condition, cond) for cond in order if cond != reference_condition]
    
    # 2. Extract the p-values based on the model type
    pvalues = []
    for _, target in box_pairs:
        
        if model_type in ['additive', 'auc']:
            # Looks for the main effect difference (e.g. C(condition, Treatment('Sparos'))[T.PC])
            term_name = f"C({x}, Treatment('{reference_condition}'))[T.{target}]"
            
        elif model_type == 'interactive':
            # Looks for the interaction/slope difference (e.g. C(condition, Treatment('Sparos'))[T.PC]:time_hr)
            term_name = f"C({x}, Treatment('{reference_condition}'))[T.{target}]:{continuous_var}"
            
        else:
            raise ValueError("model_type must be 'auc', 'additive', or 'interactive'")
            
        # Safely extract the p-value
        try:
            pval = results.pvalues[term_name]
            pvalues.append(pval)
        except KeyError:
            raise KeyError(f"Could not find p-value for '{term_name}'. Check your variable names and reference condition.")
            
    from microbiome._vendor.analysis_toolbox.visual import annotate_custom_pvalues
    # 3. Pass the automatically generated pairs and p-values to your annotator
    return annotate_custom_pvalues(
        ax=ax,
        data=data,
        x=x,
        y=y,
        order=order,
        box_pairs=box_pairs,
        pvalues=pvalues,
        text_format=text_format,
        **kwargs
    )

def analyze_microglia_time_averaged_auc(df_wide, reference_condition="Sparos"):
    """
    Calculates the Time-Averaged Area Under the Curve (AUC) for each subject 
    in a wide dataframe and runs an ANOVA to compare conditions.
    This safely handles missing data and unequal time durations.
    """
    import pandas as pd
    import numpy as np
    import scipy.integrate as integrate
    import statsmodels.formula.api as smf

    auc_records = []
    
    # Iterate through each subject in the wide dataframe
    for (condition, expt), row in df_wide.iterrows():
        # Drop missing time points for this specific subject
        valid_data = row.dropna()
        
        # We need at least 2 time points to calculate an area
        if len(valid_data) > 1:
            timepoints = valid_data.index.values.astype(float)
            counts = valid_data.values
            
            # Calculate Raw Area Under the Curve
            try:
                raw_auc = integrate.trapezoid(y=counts, x=timepoints)
            except AttributeError:
                # Fallback for older SciPy/NumPy versions
                raw_auc = np.trapz(y=counts, x=timepoints)
                
            # Convert to Time-Averaged AUC
            duration = timepoints[-1] - timepoints[0]
            
            if duration > 0: # Prevent division by zero
                time_averaged_auc = raw_auc / duration
                auc_records.append({'expt': expt, 'condition': condition, 'TimeAveragedAUC': time_averaged_auc})
            
    # Convert results into a clean dataframe
    auc_df = pd.DataFrame(auc_records)
    
    # Run the statistical test
    print(f"Running Time-Averaged AUC Analysis with '{reference_condition}' as reference...")
    formula = f"TimeAveragedAUC ~ C(condition, Treatment('{reference_condition}'))"
    model_results = smf.ols(formula, data=auc_df).fit()
    
    print(model_results.summary())
    
    return auc_df, model_results
