# Vendored from analysis_toolbox (dataorg/dataset.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import os
import pathlib
import pandas as pd
import numpy as np
from pprint import pprint
from microbiome._vendor.analysis_toolbox.janelia_helper import janelia_path
from microbiome._vendor.analysis_toolbox.utils import change_path_root, path_slash


expt_id_regex = (
    ".*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]"
)


expt_name_regex = ".*_fish[0-9][0-9]_exp[0-9][0-9].*"


date_regex = "[0-9][0-9][0-9][0-9][0-9][0-9]"


fish_regex = "fish[0-9][0-9]"


expt_regex = "exp[0-9][0-9]"


def identify_dataset(dset_path, debug=False):

    from datetime import datetime
    from microbiome._vendor.analysis_toolbox.utils import search_list_of_str
    from microbiome._vendor.analysis_toolbox.dataorg.dataset import date_regex, expt_id_regex

    # the unique identifier of any experiment
    expt_id_search = search_list_of_str(dset_path.split("/"), expt_id_regex)

    if len(expt_id_search) > 0:
        expt_id = expt_id_search[0]

    else:
        expt_name_search = search_list_of_str(dset_path.split("/"), expt_name_regex)

        if len(expt_name_search) > 0:
            expt_id = expt_name_search[0]

        else:
            expt_id = None

    if debug:
        print(f"expt_id: {expt_id}")

    # when the experiment was done
    expt_date = datetime.strptime(
        search_list_of_str(dset_path.split("/"), date_regex)[0], "%Y%m%d"
    )

    if debug:
        print(f"expt_date: {expt_date}")

    # name of the experiment, which should contain information
    # about the age of the fish, the genetic background of the
    # fish, the experimental paradigm, etc
    if expt_id is not None:

        expt_name = "_".join(expt_id.split("_")[:-2])

        expt_num_search = search_list_of_str(expt_id.split("_"), expt_regex)
        if len(expt_num_search) > 0:
            expt_num = expt_num_search[0]
        else:
            expt_num = None

        fish_num_search = search_list_of_str(expt_id.split("_"), fish_regex)
        if len(fish_num_search) > 0:
            fish_num = fish_num_search[0]
        else:
            fish_num = None

    else:
        expt_name = None

        ## still try to search for expt_num and fish_num
        expt_num_search = search_list_of_str(dset_path.split("/"), expt_regex)
        if len(expt_num_search) > 0:
            expt_num = expt_num_search[0]
        else:
            expt_num = None

        fish_num_search = search_list_of_str(dset_path.split("/"), fish_regex)
        if len(fish_num_search) > 0:
            fish_num = fish_num_search[0]
        else:
            fish_num = None

    if debug:
        print(f"expt_name: {expt_name}")
        print(f"fish_num: {fish_num}")
        print(f"expt_num: {expt_num}")

    return expt_id, expt_date, expt_name, fish_num, expt_num


def load_metadata(metadata_dir):

    try:

        print(f"\nOpening metadata file in {metadata_dir}...")

        from microbiome._vendor.analysis_toolbox.dataorg.dataset import dataset

        metadata = dataset(metadata_dir)  # metadata_dir is data dir
        metadata_pd = metadata.open_metadata(return_df=True)

        print("Metadata file loaded!")

    except Exception as e:

        print("Metadata not available...")
        print(e)

        metadata = None
        metadata_pd = None

    return metadata, metadata_pd


def open_ch0xml(im_dir):

    from microbiome._vendor.analysis_toolbox.fileio import xml2df

    return xml2df(im_dir / "ch0.xml")


def list_possible_im_dirs(
    im_dir, missing_fishxx=False, include_processed=False, verbose=False
):
    """
    List possible directories containing data.

    im_dir [str]: Pathlib directory
    """

    if missing_fishxx:
        # sometimes, I forget to put the dataset in a fishxx dir
        for subdir in im_dir.as_posix().split("/"):
            if "fish" in subdir:
                # not to be confused with the root dir which will also contain 'fish'
                if "dpf" not in subdir and "exp" not in subdir:
                    fish_dir = subdir

        if verbose:
            print(f"im_dir: {im_dir.as_posix()}")
            print(f"to replace:  {fish_dir}")

        alt_im_dir = pathlib.Path(im_dir.as_posix().replace(fish_dir + "/", ""))

    possible_dirs = [
        im_dir / "im",
        im_dir / "im_CM0",
        im_dir / "im_CM1",
        im_dir / "im_CM0_CH0",
        im_dir / "im_CM0_CH1",
        im_dir / "im_CM1_CH0",
        im_dir / "im_CM1_CH1",
    ]

    if include_processed:
        possible_dirs += [
            im_dir / "im_CM0_voluseg",
            im_dir / "im_CM1_voluseg",
            im_dir / "im_CM0_dff",
            im_dir / "im_CM1_dff",
        ]

    if missing_fishxx:
        possible_dirs += [
            alt_im_dir / "im",
            alt_im_dir / "im_CM0",
            alt_im_dir / "im_CM1",
        ]

        if include_processed:
            possible_dirs += [
                alt_im_dir / "im_CM0_voluseg",
                alt_im_dir / "im_CM1_voluseg",
                alt_im_dir / "im_CM0_dff",
                alt_im_dir / "im_CM1_dff",
            ]

    return possible_dirs


def return_sorted_weelab_tifs(src_dir):

    from microbiome._vendor.analysis_toolbox.utils import find_files

    source_files = find_files(
        src_dir, ext=".TIF", grep="*", compute_paths=True, compute_basenames=True
    )

    if len(source_files["path"]) == 0:
        source_files = find_files(
            src_dir, ext=".tif", grep="*", compute_paths=True, compute_basenames=True
        )
    assert len(source_files["path"]) > 0, f"Can't find any .TIF or .tif files in {src_dir}"

    source_paths = source_files["path"]
    source_basenames = source_files["basename"]
    source_ts = [int(basename.split("_t")[-1][:-4]) for basename in source_basenames]
    sort = np.argsort(source_ts)
    sorted_source_paths = np.array(source_paths)[sort]
    sorted_source_ts = np.array(source_ts)[sort]

    return sorted_source_paths


def return_sorted_image_files(dir_path, dir_org="ahrens"):

    from microbiome._vendor.analysis_toolbox.utils import path_slash
    from glob import glob

    if dir_org == "ahrens":

        filepaths = sorted(glob(path_slash(dir_path) + "TM*.h5"))

        if len(filepaths) == 0:
            filepaths = sorted(glob(path_slash(dir_path) + "TM*.hdf5"))
        if len(filepaths) == 0:
            print(f"Can't find any h5 files in {dir_path}")

    elif dir_org == "wee":

        from microbiome._vendor.analysis_toolbox.dataorg.dataset import return_sorted_weelab_tifs

        filepaths = return_sorted_weelab_tifs(dir_path)

    else:

        filepaths = sorted(glob(path_slash(dir_path) + f"*{fileext}"))

    return filepaths


def infer_directory_organization(dir_path):

    if "/mnt/data" in dir_path:
        return "wee"
    else:
        return "ahrens"


def ensure_array(dir_path, chunks=None, out_fmt="dask", dir_org=None, verbose=False):
    """(Lazy) load from the two most common formats: h5 and zarr"""

    if dir_org is None:

        dir_org = infer_directory_organization(dir_path)

    if isinstance(dir_path, str):

        pathname, pathext = os.path.splitext(dir_path)

        if verbose:
            print(pathname, pathext)

        if pathext == ".zarr":

            if out_fmt == "dask":
                import dask.array as da

                arr = da.from_zarr(dir_path, chunks=chunks, inline_array=True)

            elif out_fmt == "zarr":
                import zarr

                arr = zarr.open(dir_path, mode="r")

            elif out_fmt == "numpy":
                import zarr

                arr = zarr.load(dir_path)

        # assuming no file extension
        else:

            # it could still represent a zarr directory or
            # it could represent a directory containing h5 files
            from microbiome._vendor.analysis_toolbox.utils import find_files, ensure_end_in_slash

            im_paths = find_files(
                ensure_end_in_slash(dir_path),
                ext=None,
                compute_paths=True,
                verbose=False,
            )["path"]

            if len(im_paths) >= 4:
                example_im_path = im_paths[3]
            else:
                example_im_path = im_paths[0]

            filename, fileext = os.path.splitext(example_im_path)
            print(filename, fileext)

            # detect format of processed data
            # For zarr directories that are not named with .zarr at the end
            # look into the zarr directory and see if there are 0, 1, 2, ... directories
            if example_im_path.split(".")[-1] in [str(ext) for ext in range(16)]:

                if out_fmt == "dask":

                    import dask.array as da

                    arr = da.from_zarr(dir_path, chunks=chunks, inline_array=True)

                elif out_fmt == "zarr":

                    import zarr

                    arr = zarr.open(dir_path, mode="r")

                elif out_fmt == "numpy":

                    import zarr

                    arr = zarr.load(dir_path)

            elif fileext == ".h5" or fileext == ".TIF" or fileext == ".tif":

                from microbiome._vendor.analysis_toolbox.fileio import h5s_to_dask
                from fish.util.fileio import to_dask
                from microbiome._vendor.analysis_toolbox.utils import path_slash
                from glob import glob

                filepaths = return_sorted_image_files(dir_path, dir_org=dir_org)

                if fileext == ".h5":

                    if dir_org == "ahrens":
                        dset_name = "default"

                    else:

                        import h5py

                        with h5py.File(filepaths[0], "r") as hf:
                            keys = list(hf.keys())
                        if len(keys) == 1:
                            dset_name = keys[0]

                    read_kwargs = {"dataset": dset_name}

                else:
                    dset_name = None
                    read_kwargs = {}

                if out_fmt == "dask":

                    # Not sure if accurate,  but I think h5s_to_dask() reads h5
                    # files faster and with less memory than to_dask(),
                    # especially if chunks are provided
                    from microbiome._vendor.analysis_toolbox.fileio import image_files_to_dask_array

                    print(filepaths)
                    if isinstance(chunks, tuple):

                        if dset_name is None:
                            arr = image_files_to_dask_array(
                                filepaths,
                                chunks=chunks,
                            )

                        else:
                            arr = image_files_to_dask_array(
                                filepaths, chunks=chunks, dataset=dset_name
                            )

                    elif chunks == "inherit":

                        from microbiome._vendor.analysis_toolbox.fileio import return_image_structure

                        image_shape, image_chunks, image_dtype = return_image_structure(
                            filepaths[0], **read_kwargs
                        )

                        if image_chunks is None:
                            chunks = (1, None, None, None)

                        else:
                            chunks = (
                                1,
                                image_chunks[0],
                                image_chunks[1],
                                image_chunks[2],
                            )

                        arr = image_files_to_dask_array(
                            filepaths, chunks=chunks, **read_kwargs
                        )

                    else:
                        arr = image_files_to_dask_array(
                            filepaths, chunks=chunks, **read_kwargs
                        )

                        # arr = to_dask(filepaths, dset_name=dset_name) ## Old/Davis'

                elif out_fmt == "numpy":

                    from microbiome._vendor.analysis_toolbox.fileio import concatenated_h5_array

                    arr = concatenated_h5_array(filepaths)

    # if dir_path is already an array, return the array
    elif isinstance(dir_path, np.ndarray):
        arr = dir_path
    else:
        print(dir_path)
        print(type(dir_path))

    return arr


def find_im_data(
    im_dir,
    look_in_root=False,
    missing_fishxx=False,
    include_processed=False,
    verbose=True,
):
    """
    There's some inconsistency as to where the data is stored. Look
    through some logical locations.

    im_dir [str]: "root" directory containing imaging data, or
                  containing subdirectories that contain imaging data
    im_data_dirs [list]: list containing Pathlib directories
    """

    im_dirs = list_possible_im_dirs(
        im_dir, missing_fishxx=missing_fishxx, include_processed=include_processed
    )

    im_dir_exists = []
    for im_dir_ in im_dirs:
        try:
            im_dir_exists.append(im_dir_.exists())
        except:
            im_dir_exists.append(False)

    find_data = np.array(im_dirs)[np.array(im_dir_exists)]

    if verbose:

        print("===============================")
        print("Looking for imaging datasets...")
        print("===============================")

        print("Looking in these directories:")
        pprint(im_dirs)

        print("Found imaging data here:")
        pprint(find_data)

    if len(find_data) > 0:
        im_data_dirs = list(find_data)
    elif len(find_data) == 0:
        if look_in_root:
            if verbose:
                print("Didn't find any dirctories; data could be in root")
            im_data_dirs = [im_dir]
        else:
            im_data_dirs = []

    if verbose:
        print("Final determination:")
        pprint(im_data_dirs)

    return im_data_dirs


def find_relevant_vol_dirs(root_dir):

    relevant_dirs = []

    if isinstance(root_dir, str):
        import pathlib

        root_dir = pathlib.Path(root_dir)

    for postpend in ["", ".zarr", "_cit.zarr", "_cis.zarr"]:

        possible_path = root_dir.with_name(root_dir.name + postpend)

        if possible_path.exists():
            relevant_dirs.append(possible_path)

    return relevant_dirs


class dataset(object):
    def __init__(self, data_dir, data_server="", overwrite={}, verbose=False):

        self.data_dir = data_dir
        self.verbose = verbose

        if data_dir[0] == "/":
            import pathlib

            self.dat_path = pathlib.Path(data_dir)
        else:
            self.dat_path = janelia_path(
                data_dir, server=data_server
            )  # converts to pathlib.PosixPath

        # Voxel-based
        self.root_dir = self.dat_path.parents[4]
        self.home_dir = self.dat_path.parents[2]
        self.base_dir = self.dat_path.parents[0]

        # im_dir will always be a list
        # and so will be dff_dir, ds_dir and seg_dir
        self.im_dir = find_im_data(
            self.dat_path, look_in_root=False, verbose=self.verbose
        )

        from microbiome._vendor.analysis_toolbox.utils import flatten_list_of_lists

        self.zarr_dir = [
            im_dir.with_name(im_dir.name + "_dff.zarr") for im_dir in self.im_dir
        ]

        if self.zarr_dir:
            self.zarr_dir = flatten_list_of_lists(
                list(map(find_relevant_vol_dirs, self.zarr_dir))
            )

        self.dff_dir = [
            im_dir.with_name(im_dir.name + "_dff") for im_dir in self.im_dir
        ]

        if self.dff_dir:
            self.dff_dir = flatten_list_of_lists(
                list(map(find_relevant_vol_dirs, self.dff_dir))
            )

        self.ds_dir = [im_dir.with_name(im_dir.name + "_ds") for im_dir in self.im_dir]

        if self.ds_dir:
            self.ds_dir = flatten_list_of_lists(
                list(map(find_relevant_vol_dirs, self.ds_dir))
            )

        self.seg_dir = [
            im_dir.with_name(im_dir.name + "_seg") for im_dir in self.im_dir
        ]
        self.voluseg_dir = [
            im_dir.with_name(im_dir.name + "_voluseg") for im_dir in self.im_dir
        ]

        self.pro_dir = self.dat_path / "processed"
        self.ana_dir = self.dat_path / "analysis"
        self.ep_dir = self.dat_path / "ephys"

        # Print stats
        self.expt_name = self.dat_path.name
        self.fish_name = self.dat_path.parts[-2]
        self.date_name = self.dat_path.parts[-3]

        if self.verbose:

            print("Dataset configured")
            print("==================")
            print(f"Experiment name: {self.expt_name}")
            print(f"Experiment date: {self.date_name}")
            print(f"Fish ID: {self.fish_name}\n")
            print(
                "Please check that all directories are correct. Wrong "
                "directories can be replaced using the overwrite "
                "argument. Strings type paths are found by appending "
                "_str to the following:\n"
            )
            print(f"dat_path: {self.dat_path}")
            print(f"root_dir: {self.root_dir}")
            print(f"home_dir: {self.home_dir}")
            print(f"base_dir: {self.base_dir}")

            print("im_dir:")
            pprint(self.im_dir)
            print("zarr_dir:")
            pprint(self.zarr_dir)
            print("dff_dir:")
            pprint(self.dff_dir)
            print("ds_dir:")
            pprint(self.ds_dir)
            print(f"seg_dir:")
            pprint(self.seg_dir)
            print(f"voluseg_dir:")
            pprint(self.voluseg_dir)

            print(f"pro_dir: {self.pro_dir}")
            print(f"ana_dir: {self.ana_dir}")
            print(f"ep_dir: {self.ep_dir}")

            print("\n")

        df_name = self.expt_name + ".json"
        self.metadata_dir = self.dat_path / df_name

        self.root_dir_str = path_slash(self.root_dir.as_posix())
        self.home_dir_str = path_slash(self.home_dir.as_posix())
        self.ep_dir_str = path_slash(self.ep_dir.as_posix())
        self.dat_path_str = path_slash(self.dat_path.as_posix())
        self.im_dir_str = [path_slash(im_dir.as_posix()) for im_dir in self.im_dir]
        self.zarr_dir_str = [
            path_slash(zarr_dir.as_posix()) for zarr_dir in self.zarr_dir
        ]
        self.dff_dir_str = [path_slash(dff_dir.as_posix()) for dff_dir in self.dff_dir]
        self.ds_dir_str = [path_slash(ds_dir.as_posix()) for ds_dir in self.ds_dir]
        self.metadata_dir_str = self.metadata_dir.as_posix()
        self.pro_dir_str = path_slash(self.pro_dir.as_posix())
        self.ana_dir_str = path_slash(self.ana_dir.as_posix())
        self.seg_dir_str = [path_slash(seg_dir.as_posix()) for seg_dir in self.seg_dir]
        self.voluseg_dir_str = [
            path_slash(voluseg_dir.as_posix()) for voluseg_dir in self.voluseg_dir
        ]

        from microbiome._vendor.analysis_toolbox.utils import find_files

        self.im_path_str = [
            find_files(im_dir_str, compute_paths=True)["path"]
            for im_dir_str in self.im_dir_str
        ]

        self.seg_filt = [seg_dir / "cell_dff.npz" for seg_dir in self.seg_dir]
        self.seg_filt_str = [seg_filt.as_posix() for seg_filt in self.seg_filt]

    def prepare_dir(self):
        """
        Create the standard directories. For the easy creation of
        non-standard directories, see prepare_dataset_dir in
        janelia_helper.py
        """

        [im_dir.mkdir(exist_ok=True, parents=True) for im_dir in self.im_dir]
        [seg_dir.mkdir(exist_ok=True, parents=True) for seg_dir in self.seg_dir]

        self.ep_dir.mkdir(exist_ok=True, parents=True)
        self.pro_dir.mkdir(exist_ok=True, parents=True)
        self.ana_dir.mkdir(exist_ok=True, parents=True)

    def create_metadata(self, overwrite=False):
        """
        A metadata file can be generated for most imaging datasets,
        including first and foremost the parameters that went into,
        for example, the light sheet imanging experiments. Later
        decisions about the processing of the data can also be added.
        """
        print(self.im_dir)

        if isinstance(self.im_dir, list):

            metadata_dfs = [
                open_ch0xml(im_dir)
                for im_dir in self.im_dir
                if (im_dir / "ch0.xml").exists()
            ]
            print(f"Found {len(metadata_dfs)} ch0.xmls")

            # if ch0.xml file doesn't exist, create an empty dataframe to save
            if len(metadata_dfs) == 0:

                print("ch0.xml does not exists")

                df = pd.DataFrame()
                df["value"] = []
                df.at["ch0.xml_missing", "value"] = "T"

            if len(metadata_dfs) == 1:
                df = metadata_dfs[0]

            if len(metadata_dfs) == 2:

                # [!!]assumes and only compares the first two, when there could be more
                df = metadata_dfs[0][(metadata_dfs[0] == metadata_dfs[1]).values]
                unique_metadata = metadata_dfs[0][
                    (metadata_dfs[0] != metadata_dfs[1]).values
                ].index
                for uniq in unique_metadata:
                    df.loc[uniq, "value"] = (
                        metadata_dfs[0].loc[uniq, "value"]
                        + ";"
                        + metadata_dfs[1].loc[uniq, "value"]
                    )

        else:  # shouldn't apply any more, but leaving it here for posterity

            if (self.im_dir / "ch0.xml").exists():
                df = open_ch0xml(self.im_dir)

        if not self.metadata_dir.exists():
            df.to_json(self.metadata_dir)
        else:
            if overwrite:
                df.to_json(self.metadata_dir)

    def open_metadata(self, return_df=False):

        print(f"Reading metadata file {self.metadata_dir}")
        self.metadata = pd.read_json(
            self.metadata_dir, orient="columns", precise_float=True
        )

        if return_df:
            return self.metadata

    def add_to_metadata(self, key_value_dict, save_and_overwrite=False):

        if hasattr(self, "metadata"):
            pass
        else:
            self.open_metadata()

        dict_df = pd.DataFrame.from_dict(
            key_value_dict, orient="index", columns=["value"]
        )

        if save_and_overwrite:
            if any(
                new_key in list(self.metadata.index)
                for new_key in list(key_value_dict.keys())
            ):
                self.metadata.drop(key_value_dict.keys(), inplace=True)

        added_df = self.metadata.append(dict_df, verify_integrity=True)

        if save_and_overwrite:
            added_df.to_json(self.metadata_dir, double_precision=15)

        return added_df

    def add_stack_freq_to_metadata(self):

        if hasattr(self, "metadata"):
            pass
        else:
            self.open_metadata()

        if "stack_freq" in self.metadata.index:
            return None
        else:
            print("stack_freq not found. Adding...")
            self.metadata.loc["stack_freq", "value"] = ""
            self.metadata.loc["acq_dur", "value"] = ""
            self.metadata.loc["num_stacks", "value"] = ""

        if isinstance(self.im_dir, list):

            for im_dir in self.im_dir:

                if (im_dir / "Stack_frequency.txt").exists():

                    with open(im_dir / "Stack_frequency.txt", "r") as fp:

                        lines = fp.read().splitlines()

                        stack_freq = float(lines[0])
                        acq_dur = float(lines[1])
                        num_stacks = float(lines[2])

                    if self.metadata.loc["stack_freq", "value"]:
                        prepend = "; "
                    else:
                        prepend = ""

                    key_value_dict = {
                        "stack_freq": self.metadata.loc["stack_freq", "value"]
                        + f"{prepend}{stack_freq}",
                        "acq_dur": self.metadata.loc["acq_dur", "value"]
                        + f"{prepend}{acq_dur}",
                        "num_stacks": self.metadata.loc["num_stacks", "value"]
                        + f"{prepend}{num_stacks}",
                    }

                    self.metadata_pd = self.add_to_metadata(
                        key_value_dict, save_and_overwrite=True
                    )

    def init_ephys(self):
        """Figure out where the ephys files are saved"""

        print(f"Looking for ephys files in {self.ep_dir} ...\n")
        ephys_files = [f for f in pathlib.Path(self.ep_dir).glob("*chFlt")]
        print("*chFlt files found:")
        print(ephys_files)

        if len(ephys_files) == 1:
            self.ephys_file = ephys_files[0]
        else:
            for file in ephys_files:
                print(file)
            self.ephys_file = input("Ephys path: ")

            if self.ephys_file == "":
                print("ephys file not found...")
                self.ephys_file = None

        if self.ephys_file:  # if ephys file is found

            self.ephys_file_str = (
                self.ep_dir / self.ephys_file.stem
            ).as_posix() + ".npy"
            self.ephys_raw_str = self.ephys_file.as_posix()

            if os.path.exists(self.ephys_file_str):
                print("Processed ephys file exists")
                print("Using processed ephys file...")
                self.ephys_path_str = self.ephys_file_str
            elif os.path.exists(self.ephys_raw_str):
                print("Raw ephys file exists")
                self.ephys_path_str = self.ephys_raw_str

        self.ephys_num_ch = int(self.ephys_raw_str.split(".")[-1][:2])

    def search_external_dir_for_extras(
        self,
        root_dir="/nrs/ahrens/Ziqiang/Jing_Glia_project/Processed_data",
        return_pathlib=True,
        return_pathstr=False,
    ):

        extra_dir = change_path_root(
            self.dat_path, self.dat_path.parents[2].parts[-1], root_dir
        )

        possible_extra_dirs = find_im_data(
            extra_dir,
            look_in_root=True,
            missing_fishxx=True,
            include_processed=True,
            verbose=self.verbose,
        )

        from microbiome._vendor.analysis_toolbox.utils import ensure_end_in_slash

        possible_extra_dirs_str = [
            ensure_end_in_slash(possible_extra_dir.as_posix())
            for possible_extra_dir in possible_extra_dirs
        ]

        results = []
        if return_pathlib:
            results.append(possible_extra_dirs)
        if return_pathstr:
            results.append(possible_extra_dirs_str)

        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    def search_external_dir_for_stack_extras(self, im_mode):
        (
            self.possible_extras_dirs,
            self.possible_extras_dirs_str,
        ) = self.search_external_dir_for_extras(
            return_pathlib=True, return_pathstr=True
        )

        from microbiome._vendor.analysis_toolbox.utils import return_element_with_substring

        dirs_with_im_mode = return_element_with_substring(
            self.possible_extras_dirs_str, im_mode
        )
        if len(dirs_with_im_mode) == 0 and len(self.possible_extras_dirs_str) == 1:
            self.stack_extras_dir = self.possible_extras_dirs_str[0]
        else:
            self.stack_extras_dir = None

    def search_external_dir_for_segments(
        self, seg_root="/nrs/ahrens/Ziqiang/Jing_Glia_project/Processed_data"
    ):
        """Figure out where the cell/segment data are saved

        This is needed because segments could either be saved in the
        my or the segmenter's directory.

        Currently only applicable to Ziqiang's segmentation pipeline
        self.seg_zarr / self.seg_filt only applicable for Ziqiang's pipeline

        Notes
        ------
        Currently manually called when needed, but should also
        """

        # if seg_root is provided, the assumption is that the segments are in
        # a separate directory
        if seg_root:

            if self.verbose:
                pprint(f"Changing segments root directory...")
                pprint(f"Original segments directories: {self.seg_dir}")

            self.seg_root = seg_root

            (
                self.possible_seg_dirs,
                self.possible_seg_dirs_str,
            ) = self.search_external_dir_for_extras(
                root_dir=self.seg_root, return_pathlib=True, return_pathstr=True
            )

            from microbiome._vendor.analysis_toolbox.utils import return_dirs_with_contents

            self.eseg_dir = return_dirs_with_contents(
                self.possible_seg_dirs, ext="npz", grep="cell_dff"
            )

            if self.verbose:
                pprint(f"Corrected segments directories: {self.eseg_dir}")
        else:
            self.eseg_dir = []

        self.eseg_zarr = [
            self.pro_dir / "cell_raw_dff_sparse" for seg_dir in self.eseg_dir
        ]
        self.eseg_filt = [seg_dir / "cell_dff.npz" for seg_dir in self.eseg_dir]

        if self.verbose:

            print("\nSegments configured")
            print("=====================")
            print(f"seg_dir:")
            pprint(self.eseg_dir)
            print("\n")

        self.eseg_dir_str = [
            path_slash(seg_dir.as_posix()) for seg_dir in self.eseg_dir
        ]
        self.eseg_zarr_str = [seg_zarr.as_posix() for seg_zarr in self.eseg_zarr]
        self.eseg_filt_str = [seg_filt.as_posix() for seg_filt in self.eseg_filt]

    def find_segments(self):

        all_locations = self.seg_dir + self.eseg_dir + self.voluseg_dir

        valid_locations = []

        from microbiome._vendor.analysis_toolbox.utils import find_files

        for loc in all_locations:
            print(f"Looking in {loc}", end="    ")
            for grep, ext in zip(["cells_clean", "cell_dff"], ["hdf5", "npz"]):

                processed_files = find_files(
                    loc.as_posix(), ext=ext, grep=grep, compute_paths=True
                )["path"]

                if len(processed_files) > 0:
                    valid_locations.append(loc)
                    print("Found!")
                else:
                    print("Not here!")

        segments_dir = []

        if len(valid_locations) > 1:
            for ch in ["im_CM0", "im_CM1"]:
                ch_dirs = [
                    valid_loc
                    for valid_loc in valid_locations
                    if ch in valid_loc.as_posix()
                ]
                if len(ch_dirs) > 1:
                    ch_dirs = [
                        ch_dir
                        for ch_dir in ch_dirs
                        if "/limj2/" in ch_dir.as_posix()
                        or "/jing/" in ch_dir.as_posix()
                    ]
                segments_dir = segments_dir + ch_dirs
        else:
            segments_dir = valid_locations

        self.segments_dir = segments_dir

        from microbiome._vendor.analysis_toolbox.utils import path_slash

        self.segments_dir_str = [
            path_slash(segment_dir.as_posix()) for segment_dir in self.segments_dir
        ]

    def init_segments(
        self, seg_root="/nrs/ahrens/Ziqiang/Jing_Glia_project/Processed_data"
    ):

        self.search_external_dir_for_segments(seg_root=seg_root)
        self.find_segments()

        voluseg_dirs_bool = np.array(
            ["voluseg" in dir_ for dir_ in self.segments_dir_str]
        )
        if np.nansum(voluseg_dirs_bool) > 0:
            self.segments_dir_str = list(
                np.array(self.segments_dir_str)[voluseg_dirs_bool]
            )
