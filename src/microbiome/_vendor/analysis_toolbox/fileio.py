# Vendored from analysis_toolbox (fileio.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import h5py
import pandas as pd
import dask.array as da
import numpy as np
import zarr
import dask
import os
import shutil
from tqdm import tqdm_notebook as tqdm


def remove_path(path, safe=True):

    from microbiome._vendor.analysis_toolbox.utils import countdown

    if os.path.exists(path):

        if os.path.isdir(path):
            print(f"Removing dir {path}...")
            if safe:
                countdown(10, pre_msg="Okay? 10s to decide!")
            shutil.rmtree(path)
        elif os.path.isfile(path):
            print(f"Removing file {path}...")
            if safe:
                countdown(10, pre_msg="Okay? 10s to decide!")
            os.remove(path)
        else:
            print(f"Unable to remove {path}")

    else:
        print(f"Nothing to remove. The following path:\n{path}\ndoesn't exist.")


def read_tiff(filename, **kwargs):
    """
    try plugin='pil' if tifffile complains: TiffFileError: not a TIFF file
    """
    from skimage.io import imread

    return imread(filename, **kwargs)


class concatenated_h5_array(object):
    def __init__(self, h5_list):

        self.h5_list = h5_list
        print(h5_list[:10])

        # open first h5_file to figure out (1) dataset and (2) shape
        import h5py

        with h5py.File(self.h5_list[0], "r") as hf:
            keys = list(hf.keys())
            if len(keys) == 1:
                self.dset_name = keys[0]
            else:
                self.dset_name = "default"
            array_shape = hf[self.dset_name].shape

        self.shape = tuple([len(self.h5_list)] + list(array_shape))

    def __getitem__(self, slices):

        if not isinstance(slices, tuple):
            slices = tuple([slices])

        n_slices = len(slices)
        if n_slices < len(self.shape):
            slices = list(slices) + (
                [slice(None, None, None)] * (len(self.shape) - n_slices)
            )
            slices = tuple(slices)

        list_index = slices[0]
        array_index = slices[1:]

        # TODO: implement opening more than one h5 file
        h5_path = self.h5_list[list_index]

        import h5py

        with h5py.File(h5_path, "r") as hf:
            return hf[self.dset_name][array_index]


def return_h5data(h5_read, dset_names=[], out_type="dict"):
    """ """
    if out_type == "dict":
        return {
            dset_name: h5_read[dset_name][()]
            for dset_name in dset_names
            if dset_name in h5_read.keys()
        }
    elif out_type == "array":
        return np.squeeze(
            np.stack(
                {dset_name: h5_read[dset_name][()] for dset_name in dset_names}.values()
            )
        )
    else:
        return {dset_name: h5_read[dset_name][()] for dset_name in dset_names}


def readfrom_h5(filename, dset_names=[], out_type="dict"):

    import h5py

    with h5py.File(filename, "r") as hf:

        if dset_names:
            # if dataset names are included, use them
            pass
        else:
            # else, open everything
            dset_names = list(hf.keys())

        trig_dff = return_h5data(hf, dset_names, out_type=out_type)

        return trig_dff


def store_block(source, target, region, block_info=None):

    from dask.array.optimization import fuse_slice

    chunk_origin = block_info[0]["array-location"]

    slices = tuple(slice(start, stop) for start, stop in chunk_origin)

    if region:
        slices = fuse_slice(region, slices)
    target[slices] = source

    return np.expand_dims(0, tuple(range(source.ndim)))


def store_blocks(sources, targets, regions=None):
    result = []

    if isinstance(sources, dask.array.core.Array):
        sources = [sources]
        targets = [targets]

    if len(sources) != len(targets):
        raise ValueError(
            "Different number of sources [%d] and targets [%d]"
            % (len(sources), len(targets))
        )

    if isinstance(regions, tuple) or regions is None:
        regions = [regions]

    if len(sources) > 1 and len(regions) == 1:
        regions *= len(sources)

    if len(sources) != len(regions):
        raise ValueError(
            "Different number of sources [%d] and targets [%d] than regions [%d]"
            % (len(sources), len(targets), len(regions))
        )

    for source, target, region in zip(sources, targets, regions):
        out_chunks = tuple((1,) * len(c) for c in source.chunks)
        result.append(
            da.map_blocks(
                store_block, source, target, region, chunks=out_chunks, dtype="int64"
            )
        )
    return result


def read_h5(fname, dataset, region=None):
    if region is None:
        region = (slice(None),)
    with h5py.File(fname, mode="r") as fh:
        return fh[dataset][tuple(region)]


def read_h5_block(fnames, dataset, block_info):
    """
    TODO: retiring, see read_block
    """
    chunk_origin = block_info[None]["array-location"]
    outer_slice, *inner_slices = tuple(
        slice(start, stop) for start, stop in chunk_origin
    )
    to_read = fnames[outer_slice]
    return np.stack([read_h5(fname, dataset, inner_slices) for fname in to_read])


read_image_funcs = {".h5": read_h5, ".tiff": read_tiff, ".TIF": read_tiff, ".tif": read_tiff}


def read_block(fnames, block_info=None, **kwargs):
    """Read a block of image files"""

    chunk_origin = block_info[None]["array-location"]
    outer_slice, *inner_slices = tuple(
        slice(start, stop) for start, stop in chunk_origin
    )
    to_read = fnames[outer_slice]

    return load_images(to_read, **kwargs)


def load_images(fnames, **kwargs):
    ## determine file type
    import os

    filename, fileext = os.path.splitext(fnames[0])

    return np.stack([read_image_funcs[fileext](fname, **kwargs) for fname in fnames])


def infer_structure(h5, dataset):
    """
    TODO: update to return_h5_structure(), see return_image_structure()
    """
    with h5py.File(h5, libver="latest", swmr=True, mode="r") as fh:
        arr = fh[dataset]
        return arr.shape, arr.chunks, arr.dtype


def return_tif_structure(tif_path):

    from microbiome._vendor.analysis_toolbox.fileio import read_tiff

    arr = read_tiff(tif_path)

    ## return shape, chunks, dtype
    return arr.shape, None, arr.dtype


def return_image_structure(image_path, debug=False, **kwargs):

    filename, fileext = os.path.splitext(image_path)

    if fileext == ".h5" or fileext == ".hdf5":

        return infer_structure(image_path, **kwargs)

    elif fileext == ".TIF" or fileext == ".tiff" or fileext == ".tif":

        return return_tif_structure(image_path)


def image_files_to_dask_array(image_paths, chunks=None, debug=False, **kwargs):
    """
    image_paths [list]: list of file paths
    dataset [str]: dataset name needs to be provided as a kwarg if image is h5
    chunks [tuple, int or None]: if tuple, use as chunks
                                 if int, concatenate that number of h5 chunks
                                 if None, assume chunks = chunks of single h5 file
    """
    image_shape, image_chunks, image_dtype = return_image_structure(
        image_paths[0], debug=debug, **kwargs
    )

    if image_chunks == None:
        image_chunks = image_shape

    if chunks == None:
        chunks = da.core.normalize_chunks(
            (1, *image_chunks), (len(image_paths), *image_shape)
        )

    elif isinstance(chunks, int):
        chunks = da.core.normalize_chunks(
            (chunks, *image_chunks), (len(image_paths), *image_shape)
        )

    else:
        chunks = da.core.normalize_chunks(chunks, (len(image_paths), *image_shape))

    return da.map_blocks(
        read_block, image_paths, chunks=chunks, dtype=image_dtype, **kwargs
    )


def h5s_to_dask(h5s, dataset, chunks=None):
    """
    h5s [list]: list of h5 file paths
    dataset [str]: dataset name of each h5 file
    chunks [tuple, int or None]: if tuple, use as chunks
                                 if int, concatenate that number of h5 chunks
                                 if None, assume chunks = chunks of single h5 file

    TODO: retire, see image_files_to_dask_array()
    """
    h5_shape, h5_chunks, dtype = infer_structure(h5s[0], dataset)

    if h5_chunks == None:
        h5_chunks = h5_shape

    if chunks == None:
        chunks = da.core.normalize_chunks((1, *h5_chunks), (len(h5s), *h5_shape))

    elif isinstance(chunks, int):
        chunks = da.core.normalize_chunks((chunks, *h5_chunks), (len(h5s), *h5_shape))

    else:
        chunks = da.core.normalize_chunks(chunks, (len(h5s), *h5_shape))

    return da.map_blocks(read_h5_block, h5s, dataset, chunks=chunks, dtype=dtype)


def generate_list_of_empty_slices(arr, mode="append"):
    """
    Used in dask_array_to_zarr()
    """
    if mode == "append":

        results = []

        for dim in range(arr.ndim):
            results.append(slice(None, None, None))

        return results

    else:
        return [slice(None, None, None)] * arr.ndim


def dask_array_to_zarr(
    darr, target, component=None, num_sections=None, axis=None, verbose=False
):

    if num_sections is None:

        darr_save = darr
        chunks = darr_save.chunksize
        dtype = darr_save.dtype
        shape = darr_save.shape
        regions = None

    else:
        assert axis is not None

        from microbiome._vendor.analysis_toolbox.utils import dask_array_split

        darr_save, div_points = dask_array_split(
            darr, num_sections, axis, return_div_points=True
        )

        if verbose:
            display(div_points)
            display(darr_save)

        regions = []

        for sub_array, div_point in zip(darr_save, div_points):
            region = generate_list_of_empty_slices(sub_array)
            region[axis] = slice(div_point, None, None)

            regions.append(tuple(region))

        if verbose:
            display(regions)

        assert len(regions) == len(darr_save)

        shape = darr.shape
        chunks = darr_save[0].chunksize
        dtype = darr_save[0].dtype

        if verbose:
            print(f"shape: {shape}")
            print(f"chunks: {chunks}")
            print(f"dtype: {dtype}")

    target = zarr.DirectoryStore(target)

    target_zarr = zarr.open(
        store=target,
        path=component,
        chunks=chunks,
        dtype=dtype,
        shape=shape,
        overwrite=True,
    )

    if num_sections is not None:

        target_zarr = [target_zarr] * num_sections
        storage_op = [
            store_blocks(darr_save_, target_zarr_, region)
            for darr_save_, target_zarr_, region in tqdm(
                zip(darr_save, target_zarr, regions), total=len(darr_save)
            )
        ]

        from microbiome._vendor.analysis_toolbox.utils import flatten_list_of_lists

        storage_op = flatten_list_of_lists(storage_op)

    else:

        storage_op = store_blocks(darr_save, target_zarr, regions)

    return storage_op


def dask_arr_to_zarr_future(darr, zarr_path="", mode="da"):

    from microbiome._vendor.analysis_toolbox.utils import ensure_extension, ensure_filepath

    zarr_path = ensure_extension(ensure_filepath(zarr_path), "zarr")

    # TODO: might not be equivalent, I believe that dask_array_to_zarr
    # returns a list of future while da.to_zarr is a single future
    if mode == "da":
        # storage_op = da.to_zarr(darr, zarr_path, compute=False)
        storage_op = darr.to_zarr(zarr_path, compute=False)
    elif mode == "custom":
        from microbiome._vendor.analysis_toolbox.fileio import dask_array_to_zarr

        storage_op = dask_array_to_zarr(darr, zarr_path)

    return storage_op


def chunk_and_saveas_zarr(
    arr,
    chunks="auto",
    savename="",
    overwrite=False,
    debug=False,
    reload=False,
    inline_array=True,
    mode="dask",
):
    """Save in-memory array by first using dask chunking scheme, with a
    possibility of reloading.
    """
    import zarr

    from microbiome._vendor.analysis_toolbox.utils import (
        ensure_extension,
        ensure_filepath,
        get_chunksize_from_chunks,
    )

    savename = ensure_extension(ensure_filepath(savename), "zarr")

    if chunks == "auto":
        chunks = ["auto"] * arr.ndim

    if "auto" in chunks:
        chunks = get_chunksize_from_chunks(arr, chunks, mode="full")

    if debug:
        print(f"Final chunks: {chunks}")

    if overwrite or not os.path.exists(savename):
        if debug:
            print(f"Writing zarr to {savename}...")
        if mode == "dask":
            zarr.save_array(savename, arr, chunks=chunks)
        elif mode == "xarray":
            import xarray as xr

            xarr = xr.DataArray(arr)
            xarr = xarr.chunk(chunks=chunks, inline_array=inline_array)
            xarr.to_zarr(savename)

    if reload == "dask":

        if mode == "dask":
            darr = da.from_zarr(savename, inline_array=inline_array)
        elif mode == "xarray":
            darr = xr.open_zarr(
                savename, from_array_kwargs={"lock": True}
            ).__xarray_dataarray_variable__.data

        return savename, darr
    else:
        return savename


def read_zarr(store, mode="lazy"):

    import zarr

    if mode == "lazy":
        return zarr.open(store, mode="r")
    elif mode == "RAM":
        return zarr.load(store)


def fill_zarr_along_axis0(zarr_path, shape4d, idx, volume3d, dtype=None, chunks=None):
    if len(shape4d) != 4:
        raise ValueError("shape4d must be a 4D shape")
    if volume3d.ndim != 3:
        raise ValueError("volume3d must be a 3D array")
    if tuple(shape4d[1:]) != tuple(volume3d.shape):
        raise ValueError("volume3d shape must match shape4d[1:]")

    if dtype is None:
        dtype = volume3d.dtype
    if chunks is None:
        chunks = (1,) + tuple(volume3d.shape)

    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)

    zarr_array = zarr.open_array(
        zarr_path,
        mode='a',
        shape=tuple(shape4d),
        dtype=dtype,
        chunks=chunks
    )

    if zarr_array.shape != tuple(shape4d):
        raise ValueError("Existing zarr array shape does not match requested shape")

    if idx < 0 or idx >= shape4d[0]:
        raise IndexError("Index out of bounds for first axis")

    zarr_array[idx] = volume3d
    return zarr_array


def xml2df(xml):
    """
    Converts an xml file with one layer of nesting to a dataframe.
    """

    import xml.etree.ElementTree as et
    import pandas as pd

    tree = et.parse(xml)
    root = tree.getroot()
    children = list(root)  # root.getchildren()

    dict = {}

    for r in root:
        dict.update(r.attrib)

    df = pd.DataFrame.from_dict(dict, orient="index", columns=["value"])

    return df


def load_gsheet(
    sheet_name: str, worksheet: str, keyfile_name: str, debug=False
) -> pd.DataFrame:

    import gspread
    import pandas as pd
    from oauth2client.service_account import ServiceAccountCredentials

    # define the scope
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    # add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name(keyfile_name, scope)

    # authorize the clientsheet
    client = gspread.authorize(creds)

    # get the instance of the Spreadsheet
    sheet = client.open(sheet_name)

    # get the first sheet of the Spreadsheet
    sheet_instance = sheet.worksheet(worksheet)

    # get all the records of the data
    records_data = sheet_instance.get_all_records(numericise_ignore=["all"])

    # convert the json to dataframe
    records_df = pd.DataFrame.from_dict(records_data)

    if debug:

        # view the top records
        records_df.head()

    return records_df


def make_local_copy(src_path, dest_dir):

    from microbiome._vendor.analysis_toolbox.utils import change_root_dir_in_path
    
    temp_local_path = change_root_dir_in_path(src_path, dest_dir)

    if os.path.exists(temp_local_path):
        shutil.rmtree(temp_local_path)

    if not os.path.exists(temp_local_path):
        shutil.copytree(src_path, temp_local_path)

    return temp_local_path
