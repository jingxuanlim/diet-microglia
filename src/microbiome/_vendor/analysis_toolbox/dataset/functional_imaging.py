# Vendored from analysis_toolbox (dataset/functional_imaging.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


from microbiome._vendor.analysis_toolbox.dataset.expt import expt
import os
import numpy as np
import pandas as pd
import dask.array as da
import matplotlib.pyplot as plt
import warnings
from pprint import pprint
from tqdm.auto import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
from microbiome._vendor.analysis_toolbox.utils import rgetattr


tqdm_notebook.pandas(desc="Processing...")


def load_func_im(
    data_dir,
    data_server,
    ephys="raw",
    im_raw=False,
    im_pro=False,
    segmented=False,
    align=True,
    channels="all",
    dataset_type="spim",
    project="statemod",
):
    """
    dataset_type [str]: 'spim', 'resonant2P', 'old2P'
    """

    from microbiome._vendor.analysis_toolbox.dataorg.dataset import dataset

    dset = dataset(data_dir, data_server, verbose=True)

    if ephys:
        dset.init_ephys()

    if not os.path.exists(dset.metadata_dir_str):
        dset.prepare_dir()
        dset.create_metadata()
        dset.add_stack_freq_to_metadata()

    if segmented:
        dset.init_segments(
            # seg_root=''
        )

    from microbiome._vendor.analysis_toolbox.dataset.functional_imaging import configure_func_im

    spim_dset = configure_func_im(
        dset,
        ephys=ephys,
        im_raw=im_raw,
        im_pro=im_pro,
        segmented=segmented,
        channels=channels,
        align=align,
        dataset_type=dataset_type,
        project=project,
    )
    spim_dset.init()

    return dset, spim_dset


def configure_func_im(
    dset,
    ephys="raw",
    dataset_type="spim",
    im_raw=False,
    im_pro=False,
    segmented=False,
    align=True,
    channels="all",
    project="statemod",
):
    """Put directories in dset in the right places of the config
    ephys [str/bool]: 'raw', 'processed', True, False
    im_raw [bool]: "spim", "twophoton", False
    im_pro [bool]: True of False
    segmented [str/bool]: True, False,
                          'load_filtered', 'load_unfiltered',
                          'load_filtered_dff', 'load_segment_loc'
    project [str]: 'statemod', 'osc'
    """

    from microbiome._vendor.analysis_toolbox.utils import any_substring_in_string

    ephys_ops = {}
    twophoton_ops = {}
    spim_ops = {}

    if (
        dataset_type == "resonant2P"
        or dataset_type == "old2P"
        or dataset_type == "nikon"
    ):

        if im_raw:
            twophoton_ops["im_path"] = [
                path_str
                for im_dir_str in dset.im_path_str
                for path_str in im_dir_str
                if any_substring_in_string(path_str, channels) or channels == "all"
            ]

        if im_pro:
            twophoton_ops["impro_dir"] = [
                zarr_dir_str
                for zarr_dir_str in dset.zarr_dir_str
                if any_substring_in_string(zarr_dir_str, channels) or channels == "all"
            ]

        if im_raw or im_pro:
            twophoton_ops["metadata"] = dset.dat_path_str
            twophoton_ops["debug"] = False

    if dataset_type == "spim":

        if im_raw:
            spim_ops["im_dir"] = [
                dir_str
                for dir_str in dset.im_dir_str
                if any_substring_in_string(dir_str, channels) or channels == "all"
            ]

        if im_pro:

            spim_ops["impro_dir"] = {}

            for path in dset.dff_dir_str:
                if any_substring_in_string(path, channels) or channels == "all":
                    name = path.split("/")[-2]
                    if "." in name:
                        name = name.split(".")[0]
                    spim_ops["impro_dir"][name] = path

            for path in dset.ds_dir_str:
                if any_substring_in_string(path, channels) or channels == "all":
                    name = path.split("/")[-2]
                    if "." in name:
                        name = name.split(".")[0]
                    spim_ops["impro_dir"][name] = path

            print(f"Added processed directories: {spim_ops['impro_dir']}")

        if segmented:
            spim_ops["segmented_root"] = [
                dir_str
                for dir_str in dset.segments_dir_str
                if any_substring_in_string(dir_str, channels) or channels == "all"
            ]

            spim_ops["segmented_im_dir"] = [
                dir_str
                for dir_str in dset.im_dir_str
                if any_substring_in_string(dir_str, channels) or channels == "all"
            ]

            spim_ops["external_segmented_root"] = [
                dir_str
                for dir_str in dset.possible_seg_dirs_str
                if any_substring_in_string(dir_str, channels) or channels == "all"
            ]

            spim_ops["mode"] = segmented

        if im_raw or im_pro or segmented:
            spim_ops["metadata"] = dset.dat_path_str
            spim_ops["debug"] = False

    if ephys:

        if project == "statemod":

            from statemod.givingup_helper import channel_labels_dict

            channel_labels = channel_labels_dict[str(dset.ephys_num_ch)]

        elif project == "osc":

            if dataset_type == "spim":

                from oscillations.osc_helper import (
                    SPIM_channel_labels as channel_labels,
                )

            elif dataset_type == "resonant2P":

                from oscillations.osc_helper import (
                    resonant2P_channel_labels as channel_labels,
                )

            elif dataset_type == "old2P":

                from oscillations.osc_helper import (
                    old2P_channel_labels as channel_labels,
                )

        if ephys == "raw":

            ephys_ops = {
                "ephys_file": dset.ephys_raw_str,
                "channel_labels": channel_labels,
                "metadata": dset.dat_path_str,
            }

        elif ephys == "processed":

            ephys_ops = {
                "ephys_file": dset.ephys_file_str,
                "channel_labels": channel_labels,
                "metadata": dset.dat_path_str,
            }

        else:  # autodetect

            import os

            if os.path.isfile(dset.ephys_file_str):
                ephys_file = dset.ephys_file_str
            else:
                ephys_file = dset.ephys_raw_str

            ephys_ops = {
                "ephys_file": ephys_file,
                "channel_labels": channel_labels,
                "metadata": dset.dat_path_str,
            }

    print("ephys_ops:")
    pprint(ephys_ops)
    print("twophoton_ops:")
    pprint(twophoton_ops)
    print("spim_ops:")
    pprint(spim_ops)

    print("\nFunctional imaging experiment configured!")
    print("=========================================\n")

    return func_im(
        spim_ops=spim_ops,
        twophoton_ops=twophoton_ops,
        ephys_ops=ephys_ops,
        align=align,
        metadata=dset.dat_path_str,
        debug=False,
    )


class func_im(expt):
    def __init__(
        self,
        spim_ops={},
        twophoton_ops={},
        ephys_ops={},
        align=True,
        metadata="",
        debug=False,
    ):

        super().__init__(
            spim_ops=spim_ops,
            twophoton_ops=twophoton_ops,
            ephys_ops=ephys_ops,
            metadata_dir=metadata,
            debug=debug,
        )

        self.align = align

    def init(self):

        print("Loading functional imaging experiment! ...\n")

        ## Determine dataset
        self.identify_dataset()

        if self.debug:
            display(self.metadata_pd)

        ## Determine rig
        self.determine_rig()

        # Load ephys if ephys_ops dict is not empty
        if self.ephys_ops:

            self.load_ephys()

            self.ep_israw = self.ep.ephys_file.split(".")[-1][-5:] == "chFlt"

            if self.rig == "spim1" or self.rig == "spim2":

                if "ch0.xml_missing" in self.metadata_pd.index:
                    if self.metadata_pd.loc["ch0.xml_missing", "value"] == "F":
                        self.ep.extract_spim_image_times()
                else:
                    self.ep.extract_spim_image_times()

            elif self.rig == "resonant2p":

                self.ep.extract_twophoton_image_times()

        # Load imaging if spim_ops dict is not empty
        if self.spim_ops:
            self.load_spim()

        if self.twophoton_ops:
            self.load_twophoton()

        # Variables set
        # self.num_planes
        self.compute_numplanes()

        # self.im_eq: whether the processed images are congruent
        # self.seg_eq: whether the segments are congruent
        # self.im_dsets: all non-segmented datasets
        # self.all_dsets: all datasets, segmented and non-segmented
        self.im_dsets = self.gather_im_data()
        self.seg_dsets = self.gather_seg_data()
        self.check_congruence()

        # self.image_starts: boolean array of stack or image start times
        # self.image_starttimes: stack or image start times
        # self.im_num: number of stack or images
        # self.im_intervals: stack of image intervals
        # self.nstacks: total number of stacks or images
        # nstack: number of stacks or images
        # ephys_rate: ephys acquisition rate
        # duration: duration of experiment
        # im_rate: image acqusition rates
        self.all_dsets = self.gather_imaging_data()

        # when both imaging and ephys datasets are present, align them
        if self.all_dsets and self.ephys_ops and self.align:

            self.align_ephys_im()

        else:

            try:
                self.im_rate = float(
                    self.metadata_pd.loc["stack_freq"].value.split(";")[0]
                )
            except:
                print("metadata_pd.loc['stack_freq'] not found")

            try:
                self.nstack = int(
                    float(self.metadata_pd.loc["num_stacks"].value.split(";")[0])
                )
            except:
                print("metadata_pd.loc['num_stacks'] not found")

            try:
                self.duration = float(
                    self.metadata_pd.loc["acq_dur"].value.split(";")[0]
                )
            except:
                print("metadata_pd.loc['acq_dur'] not found")

            if hasattr(self, "ep"):
                self.ephys_rate = self.ep.rate
            else:
                print("ephys data not loaded")
                self.ephys_rate = 6000

        self.find_segments_and_their_dfs()
        self.put_segments_in_df()

        for im_mode in getattr(self, "im_modes"):

            if "seg" in im_mode:
                getattr(self, im_mode).segment_dual_color()

        # concatenate single plane data from ttyx to tyx
        for key, value in self.__dict__.items():

            if key.endswith("_raw") or key.endswith("_ds") or key.endswith("_dff"):

                if value.metadata_pd is not None:
                    if "hs_single_plane" in value.metadata_pd.index:
                        if value.metadata_pd.loc["hs_single_plane"].value == "T":
                            setattr(value, "data", da.concatenate(value.data, axis=0))

        # compute the time axis
        # self.t_axis
        # self.im_rate if it is not already available
        self.compute_taxis()

        # Downsampling
        self.downsample_factor = None

        # Interactive variables
        # Derived ephys
        self.swim = None
        self.swim_power = None
        self.swim_starts = None
        self.swim_stops = None
        self.swimming = None

        # Chopped ephys
        self.chopped_swim = None
        self.chopped_swim_power = None
        self.chopped_swim_starts = None
        self.chopped_swim_stops = None
        self.chopped_swimming = None

        # Downsampled ephys
        self.ds_swim = None
        self.ds_swimpower = None
        self.ds_swimstarts = None
        self.ds_swimstops = None
        self.ds_swimming = None

        # Indexed in the downsampled domain
        self.ds_swim_start_idx = None
        self.ds_swim_stop_idx = None

        # for fixing trial numbers with epoch information
        self.ch7_fixed = False  # will become True when it is fixed

    def identify_dataset(self, debug=False):

        from microbiome._vendor.analysis_toolbox.dataorg.dataset import identify_dataset

        self.expt_id, self.expt_date, self.expt_name, self.fish_num, self.expt_num = (
            identify_dataset(self.metadata_dir, debug=debug)
        )

    def determine_rig(self, debug=False):
        """
        TODO: Determination of ephys, old2p and resonant2p rigs ineffective
        """

        if self.metadata_pd is not None:

            if "detection_objective" in self.metadata_pd.index:
                if (
                    self.metadata_pd.loc["detection_objective", "value"]
                    == "Nikon 16x/0.80;Zeiss 20x/1.00"
                ):
                    self.rig = "spim2"
                else:
                    self.rig = "spim1"
            else:
                self.rig = "ephys"
                self.rig = "resonant2p"
        else:
            self.rig = "nikon2p"

    def find_segments_and_their_dfs(self):

        from microbiome._vendor.analysis_toolbox.utils import attributes_contain

        im_modes = attributes_contain(self, contains=["im_CM0", "im_CM1"])

        df_names = []
        self.seg_df_names = {}
        for im_mode in im_modes:

            if im_mode.endswith("_seg") or im_mode.endswith("_voluseg"):
                df_name = "seg_df"
            elif im_mode.endswith("_dff") or im_mode.endswith("_ds"):
                df_name = "voxel_df"
            else:
                df_name = None

            if df_name is not None:
                df_names.append(df_name)
                self.seg_df_names[im_mode] = df_name

        self.im_modes = im_modes
        self.df_names = df_names

        print(f"Found {self.im_modes} {self.df_names}")

    def put_segments_in_df(self):

        # put segments in a dataframe
        # TODO: refactor into function in seg_helper
        for key, value in self.__dict__.items():

            if key.endswith("_seg") or key.endswith("_voluseg"):

                value.put_segments_in_df()

                if hasattr(
                    getattr(self, key), self.seg_df_names[key]
                ) and os.path.exists(self.procpath + f"{key}_cell-type.pkl"):
                    cell_type_df = pd.read_pickle(
                        self.procpath + f"{key}_cell-type.pkl"
                    )
                    getattr(getattr(self, key), self.seg_df_names[key])[
                        "ID"
                    ] = cell_type_df

    def put_voxels_in_df(self):

        for key, value in self.__dict__.items():

            if key.endswith("_ds") or key.endswith("_dff"):

                value.put_voxels_in_df()

    def compute_numplanes(self):

        if self.rig == "spim1" or self.rig == "spim2":
            if not "ch0.xml_missing" in self.metadata_pd.index:

                print("ch0.xml found")
                if "hs_single_plane" in self.metadata_pd.index:
                    if self.metadata_pd.loc["hs_single_plane"].value == "T":
                        print("ch0.xml: single plane")
                        self.num_planes = 1
                    elif self.metadata_pd.loc["hs_single_plane"].value == "F":
                        self.num_planes = (
                            int(self.metadata_pd.loc["planes"].value.split("-")[-1]) + 1
                        )
                        print(f"ch0.xml: {self.num_planes} planes")

                else:
                    print(
                        "ch0.xml: hs_single_plane not found, pull values from 'planes'"
                    )
                    self.num_planes = (
                        int(self.metadata_pd.loc["planes"].value.split("-")[-1]) + 1
                    )

        elif self.rig == "resonant2p":

            self.num_planes = 1

    def compute_taxis(self, recompute=False):
        """Computes the t_axis in seconds

        if ephys data exist, use stack onset times to generate time stamps
        else, use average stack frequency to create a fake time line

        """

        if recompute:
            del self.im_rate

        if hasattr(self, "ep"):

            # if "ch0.xml_missing" in self.metadata_pd.index:

            if "hs_single_plane" in self.metadata_pd.index:
                if self.metadata_pd.loc["hs_single_plane"].value == "T":

                    self.t_axis = self.ep.planestart_times / self.ep.rate

                elif self.metadata_pd.loc["hs_single_plane"].value == "F":

                    self.t_axis = self.ep.stackstart_times / self.ep.rate
            else:
                if self.num_planes == 1:
                    self.t_axis = self.ep.planestart_times / self.ep.rate
                else:
                    self.t_axis = self.ep.stackstart_times / self.ep.rate

        elif (
            self.metadata_pd is not None
        ):  # TODO: will throw an error when both ephys and im information aren't available

            if "stack_freq" in self.metadata_pd.index:
                print("ephys data not found... using averages to construct t_axis...")

                stack_freq = float(
                    self.metadata_pd.loc["stack_freq"].value.split(";")[0]
                )

                if "hs_single_plane" in self.metadata_pd.index:

                    if self.metadata_pd.loc["hs_single_plane"].value == "F":

                        self.im_rate = stack_freq
                        numacq = int(
                            float(
                                self.metadata_pd.loc["num_stacks"].value.split(";")[0]
                            )
                        )

                    elif self.metadata_pd.loc["hs_single_plane"].value == "T":

                        planes_per_stack = (
                            int(
                                self.metadata_pd.loc["planes"]
                                .value.split(";")[0]
                                .split("-")[1]
                            )
                            + 1
                        )
                        self.im_rate = stack_freq * planes_per_stack
                        num_stacks = int(
                            float(
                                self.metadata_pd.loc["num_stacks"].value.split(";")[0]
                            )
                        )
                        numacq = num_stacks * planes_per_stack

                else:  # assume that if that metadata is not available, then it's NOT HS single plane

                    self.im_rate = stack_freq
                    numacq = int(
                        float(self.metadata_pd.loc["num_stacks"].value.split(";")[0])
                    )

                im_interval = 1 / self.im_rate
                duration = numacq * im_interval
                self.t_axis = np.arange(0, duration, im_interval)[:numacq]

    def gather_im_data(self):
        """Find all imaging data"""

        import collections
        from copy import deepcopy

        # Find all image-related datasets
        im_dsets = collections.OrderedDict()

        for key, value in self.__dict__.items():

            if ("_raw" in key or "_ds" in key or "_dff" in key) and "CM" in key:

                print(f"{key} detected")
                im_dsets[key + ".data"] = (
                    value.data
                )  # recently, im_dsets[key+'.data'] = value.data
                # I might have broken something

        return im_dsets

    def gather_seg_data(self):
        """Find all segmented data"""

        import collections

        seg_dsets = collections.OrderedDict()

        # transpose from ncells x ntimepoints to ntimepoints x ncells
        # in preparation for aligning the segments and imaging data later
        for key, value in self.__dict__.items():

            if key.endswith("_seg"):

                print(f"{key} detected")

                if hasattr(value, "Cmpnt_F"):
                    seg_dsets[key + ".Cmpnt_F"] = value.Cmpnt_F.transpose()
                if hasattr(value, "fCmpnt_F"):
                    seg_dsets[key + ".fCmpnt_F"] = value.fCmpnt_F.transpose()
                if hasattr(value, "cmpnt_dff"):
                    seg_dsets[key + ".cmpnt_dff"] = value.cmpnt_dff.transpose()

            if key.endswith("_voluseg"):

                print(f"{key} detected")

                if hasattr(value, "cell_baseline"):
                    seg_dsets[key + ".cell_baseline"] = value.cell_baseline.transpose()
                if hasattr(value, "cell_timeseries"):
                    seg_dsets[key + ".cell_timeseries"] = (
                        value.cell_timeseries.transpose()
                    )
                if hasattr(value, "cell_timeseries_raw"):
                    seg_dsets[key + ".cell_timeseries_raw"] = (
                        value.cell_timeseries_raw.transpose()
                    )
                if hasattr(value, "cell_dff"):
                    seg_dsets[key + ".cell_dff"] = value.cell_dff.transpose()

        return seg_dsets

    def gather_imaging_data(self):
        """Find all imaging and segmented data and put them together in a single dictionary"""

        im_dsets = self.gather_im_data()
        seg_dsets = self.gather_seg_data()

        return {**im_dsets, **seg_dsets}

    def check_congruence(self):

        print(f"Checking congruence between multiple imaging datasets...")

        # if more than one image datasets is detected, check their congruence
        if len(self.im_dsets) > 1:
            self.im_eq = self.check_congruence_im()
            if not self.im_eq:
                raise Exception("Processed image data not congruent!")

        # if both segmented and at least one imaging dataset exist
        if np.logical_and(self.im_dsets, self.seg_dsets):
            self.seg_eq = self.check_congruence_seg()
            if not self.seg_eq:
                raise Exception("Segmented data not congruent!")

    def check_congruence_im(self):
        """Check if processed data is congruent

        PIPELINE (5Ba)
        Determine if the processed data is accurately derived from the
        raw data. The processed data is commonly downsampled in x and
        y, so check for coherence in the number of planes (i.e. z) and
        the number of stacks (i.e. t).

        """

        from microbiome._vendor.analysis_toolbox.utils import checkEqual3, group_list_of_strings

        num_parts = len(list(self.im_dsets.keys())[0].split("_"))
        print(num_parts)

        im_groups = group_list_of_strings(
            list(self.im_dsets.keys()),
            number_of_substring_blocks=num_parts - 2,
            debug=True,
        )

        equality = []
        for im_group in im_groups:

            ## pull out the t and z dimensions
            im_dset_shapes = [self.im_dsets[im].shape[:2] for im in im_group]

            print(im_dset_shapes)

            equality.append(checkEqual3(im_dset_shapes))

        return np.all(equality)

    def check_congruence_seg(self):
        """Check if segmented data is congruent

        PIPELINE (5Bb)
        Determine if the segmented data is accurately derived from the
        raw data. Since we commonly downsample in x and y, check for
        coherence in the number of planes (i.e. z) and the number of
        stacks (i.e. t).

        """

        checks = []

        for key, value in self.__dict__.items():

            if key.endswith("_seg"):

                # check nummber of time points with total number of stacks in the imaging data
                # use the "first key" because it doesn't matter -- they're congruent
                if hasattr(value, "Cmpnt_F"):
                    checks.append(
                        value.Cmpnt_F.shape[1]
                        == self.im_dsets[list(self.im_dsets.keys())[0]].shape[0]
                    )
                if hasattr(value, "fCmpnt_F"):
                    checks.append(
                        value.fCmpnt_F.shape[1]
                        == self.im_dsets[list(self.im_dsets.keys())[0]].shape[0]
                    )
                elif hasattr(value, "cmpnt_dff"):
                    checks.append(
                        value.cmpnt_dff.shape[1]
                        == self.im_dsets[list(self.im_dsets.keys())[0]].shape[0]
                    )

                # check number of planes
                checks.append(
                    value.volume_mean.shape[0]
                    == self.im_dsets[list(self.im_dsets.keys())[0]].shape[1]
                )

            if key.endswith("_voluseg"):

                # check nummber of time points with total number of stacks in the imaging data
                # use the "first key" because it doesn't matter -- they're congruent
                if hasattr(value, "cell_timeseries"):
                    checks.append(
                        value.cell_timeseries.shape[1]
                        == self.im_dsets[list(self.im_dsets.keys())[0]].shape[0]
                    )
                if hasattr(value, "cell_timeseries_raw"):
                    checks.append(
                        value.cell_timeseries_raw.shape[1]
                        == self.im_dsets[list(self.im_dsets.keys())[0]].shape[0]
                    )
                if hasattr(value, "cell_baseline"):
                    checks.append(
                        value.cell_baseline.shape[1]
                        == self.im_dsets[list(self.im_dsets.keys())[0]].shape[0]
                    )
                if hasattr(value, "cell_dff"):
                    checks.append(
                        value.cell_dff.shape[1]
                        == self.im_dsets[list(self.im_dsets.keys())[0]].shape[0]
                    )

                # check number of planes
                checks.append(
                    value.volume_mean.shape[0]
                    == self.im_dsets[list(self.im_dsets.keys())[0]].shape[1]
                )

        return np.logical_and.reduce(checks)

    def align_ephys_im(self):
        """
        PIPELINE (5C) -- wrapper
        A wrapper for `self.match_ephys_im` that isn't necessary
        if dask allows setitem, allowing the original images to be
        changed (instead of relying on the creation of a new object).

        Some discussion here:
        - https://github.com/dask/dask/issues/2000 (set item)
        - https://github.com/dask/dask/issues/5199 (in place operators)
        """

        print("Aligning ephys and imaging...")
        timeseries = list(self.all_dsets.values())

        aligned = self.match_ephys_im(timeseries=timeseries)

        # once the time series are aligned, set them back as attributes of the func_im class
        for i, (key, value) in enumerate(self.all_dsets.items()):

            # if ('_raw' in key  \
            #    or '_ds' in key \
            #    or '_dff' in key)\
            #    and (('cell' not in key) and ('cmpnt' not in key))\
            #    and 'CM' in key:
            #     key +='.data'

            print("self." + key + " = aligned[%i]" % i)
            exec("self." + key + " = aligned[%i]" % i)
            # setattr(self, key, aligned[i])

        # transpose back
        for key, value in self.__dict__.items():

            if key.endswith("_seg"):

                if hasattr(value, "Cmpnt_F"):
                    setattr(value, "Cmpnt_F", getattr(value, "Cmpnt_F").transpose())
                if hasattr(value, "fCmpnt_F"):
                    setattr(value, "fCmpnt_F", getattr(value, "fCmpnt_F").transpose())
                if hasattr(value, "cmpnt_dff"):
                    setattr(value, "cmpnt_dff", getattr(value, "cmpnt_dff").transpose())

            elif key.endswith("_voluseg"):

                if hasattr(value, "cell_timeseries"):
                    setattr(
                        value,
                        "cell_timeseries",
                        getattr(value, "cell_timeseries").transpose(),
                    )
                if hasattr(value, "cell_timeseries_raw"):
                    setattr(
                        value,
                        "cell_timeseries_raw",
                        getattr(value, "cell_timeseries_raw").transpose(),
                    )
                if hasattr(value, "cell_baseline"):
                    setattr(
                        value,
                        "cell_baseline",
                        getattr(value, "cell_baseline").transpose(),
                    )
                if hasattr(value, "cell_dff"):
                    setattr(value, "cell_dff", getattr(value, "cell_dff").transpose())

    def match_ephys_im(self, timeseries=[]):
        """
        PIPELINE (5C)
        First determine how many images were recorded as taken in the
        ephys, then compare it with the images actually taken in the
        imaging dataset.

        If there are the same number of images, they are matched. If
        they are not matching, we have to align them by either removing
        frames from the ephys or the imaging data.

        Once they are matching, or if they already matched from the
        start, remove the last stack from both anyway because the last
        one could sometimes be problematic.
        """

        print("\n==========================")
        print("Aligning ephys and im data")
        print("==========================")

        self.compute_imagingtimes_(timeseries=timeseries)

        self.check_align_(timeseries=timeseries)

        ## Remove values from the END ##
        while np.logical_not(self.aligned):
            print("Cleaning up artefacts...")
            timeseries = self.cleanup_artefacts_(timeseries=timeseries)

        # if ephys and imaging are aligned and raw imaging file is
        # is used, remove last frame
        # if this is a processed imaging file, last frame has already
        # been removed, so don't do it again
        ## Remove values from the END ##
        if self.aligned and self.ep_israw:
            print("Removing last frame...")
            timeseries = self.remove_lastframe_(timeseries=timeseries)

        # print out final shapes for sainty check
        print("Final image shapes. Check that they have the same " "number of stacks: ")
        print([series.shape for series in timeseries])

        print("Computed imaging rate: %f" % self.im_rate)

        return timeseries

    def compute_imagingtimes_(self, timeseries=[]):
        """
        PIPELINE (5Ca) -- wrapper
        """

        # if imaging data exists, look at imaging data
        if self.im_dsets:
            t = timeseries[0].shape[0]
        elif self.seg_dsets:
            t = timeseries[0].shape[1]
        else:
            raise ValueError(
                "Cannot determine number of time points from imaging or segmentation data..."
            )

        z = self.num_planes

        self.compute_imagingtimes(t, z)

    def compute_imagingtimes(self, t, z):
        """
        PIPELINE (5Ca)
        Compute imaging times. Depending  on the shape of the image
        stacks, which will determine whether this a high speed single
        plane or multiplane data, the speed of the regularly acquired
        image object will be calculated. In the case of multiplane data,
        it's stacks/s. In the case of single plane data, it's plane/s.

        Returns
        --------
        image_starts: boolean array of stack or image start times
        image_starttimes: stack or image start times
        im_num: number of stack or images
        im_intervals: stack of image intervals
        """

        self.nstacks = t

        if z == 1:

            print("Detected high speed single plane data")
            self.image_starts = self.ep.plane_starts

        else:

            print("Detected multi-plane data")
            self.image_starts = self.ep.stack_starts

            if np.sum(self.ep.stack_starts) * z != np.sum(self.ep.plane_starts):

                warnings.warn("Number of stacks and planes don't match!")
                print(
                    f"Number of stacks (stackstarts): {len(self.ep.stackstart_times)}"
                )
                print(
                    f"""Number of stacks (planestarts): {len(self.ep.planestart_times)//z}
                      Images left (0 is good): {len(self.ep.planestart_times)%z}"""
                )

        self.image_starttimes = np.where(self.image_starts)[0]
        self.im_num = self.compute_framenum(self.image_starts)
        self.im_intervals = np.diff(self.image_starttimes)

        if self.debug:
            # Average imaging times only make sense when imaging rate
            # is regular. The imaging intervals will give us an idea
            # of whether that is the case.
            print("Imaging intervals: %s" % np.unique(self.im_intervals))

            acq_fig, acq_ax = plt.subplots(
                5, 1, figsize=(9, 8), sharex=True, facecolor="w"
            )

            acq_ax[0].plot(self.ep.im_times)
            acq_ax[0].set_title("Imaging times")

            acq_ax[1].plot(self.ep.stack_starts)
            acq_ax[1].set_title("Multi-plane: stack onset times (no data acquired)")

            acq_ax[2].plot(self.ep.plane_starts)
            acq_ax[2].set_title("Multi-plane: Plane acquisition times")

            acq_ax[3].plot(self.ep.stack_plane_starts)
            acq_ax[3].set_title("HS single-plane: Plane acquisition times")

            acq_ax[4].plot(self.image_starts)
            acq_ax[4].set_title("Image starts used")

            try:
                acq_ax[0].set_xlim(0, np.where(self.ep.stack_starts)[0][2])
            except:
                pass

            acq_fig.tight_layout()

    def compute_framenum(self, image_starts):

        frame_n = -1
        frame_num = np.zeros(image_starts.shape)

        for t, trig in np.ndenumerate(image_starts.astype("int")):

            if trig == 1:

                frame_n += 1

            frame_num[t] = frame_n

        return frame_num

    def plot_framenum(self):

        sfig, sax = plt.subplots(2, 1, figsize=(9, 4), sharex=True)

        t_axis = np.linspace(0, len(self.im_num), num=len(self.im_num))

        sax[0].plot(t_axis, self.image_starts)
        sax[1].plot(t_axis, self.im_num)

        end_im = 3

        sax[0].set_xlim([self.image_starttimes[0], self.image_starttimes[end_im]])
        sax[1].set_ylim([-1, end_im])

        sax[0].set_title("Image starts")
        sax[1].set_ylabel("Ephys frame")
        sax[1].set_title("Frame number")

        sfig.tight_layout()

        return sfig, sax

    def check_align(self, t):
        """
        PIPELINE (5Cb)
        Check if the number of images recorded in the ephys is the
        same as the actual number of images acquired.

        Returns
        --------
        aligned: True/False
        """

        if self.image_starts.sum() != t:
            print(
                "Number of images recorded after "
                "alignment (%i) and number of images actually taken (%i) "
                "don't match. Trying to fix that now..." % (self.image_starts.sum(), t)
            )

            self.aligned = False

        else:
            print(
                "Datasets are aligned! Number of images record after alignment = "
                "Number of images actually taken = %i" % self.image_starts.sum()
            )

            # if ephys and imaging are aligned, compute rates
            self.compute_rates(t)

            self.aligned = True

    def check_align_(self, timeseries=[]):

        t = timeseries[0].shape[0]

        self.check_align(t)

    def compute_rates(self, t):
        """
        Compute ephys and imaging rates

        Returns
        -------
        nstack: number of stacks or images
        ephys_rate: ephys acquisition rate
        duration: duration of experiment
        im_rate: image acqusition rates

        """

        self.nstacks = t
        self.ephys_rate = self.ep.rate
        self.duration = self.ep.t / self.ephys_rate
        self.im_rate = self.nstacks / self.duration

    def cleanup_artefacts_(self, timeseries=[]):
        """
        PIPELINE (5Cc) -- wrapper
        This assumes that the first recorded stack in the ephys file
        is indeed the first stack that is taken and that the ends of
        the experiment is problematic.
        """

        num_lsim = timeseries[0].shape[0]
        num_epim = self.image_starts.sum()

        n_imdiff = num_epim - num_lsim
        print(f"Number of light sheet images: {num_lsim}")
        print(f"Number of ephys images: {num_epim}")

        if n_imdiff > 0:

            print("More images in ephys. Truncating ephys...")

            diff_idx = self.image_starttimes[-n_imdiff]
            self.ep.replace_ephys(self.ep.ep[:, :diff_idx])
            if self.rig == "resonant2p":
                self.ep.extract_twophoton_image_times()
            elif "spim" in self.rig:
                self.ep.extract_spim_image_times()

            self.compute_imagingtimes_(timeseries=timeseries)

        elif n_imdiff < 0:

            print("More images in imaging. Truncating imaging...")
            # this is unlikely, and could be due to accidentally starting next
            # imaging experiment, before stopping the ephys recording

            # the following is ideal and works on numpy arrays, but it
            # doesn't work with dask arrays
            # for i,series in enumerate(timeseries):
            #     timeseries[i][:] = series[:n_imdiff]

            print([len(series) for series in timeseries])
            timeseries = [
                series[:n_imdiff]
                for i, series in enumerate(timeseries)
                if series is not None
            ]
            print([len(series) for series in timeseries])

        self.check_align_(timeseries)

        return timeseries

    def remove_lastframe_(self, timeseries=[]):
        """
        PIPELINE (5Cd) -- wrapper
        There could be the same number of images in both ephys and imaging but
        the ends are not aligned.
        """

        print("Ephys and imaging aligned; remove last frame from both...")

        # the following is ideal and works on numpy arrays, but it
        # doesn't work with dask arrays
        # for i,series in enumerate(timeseries):
        #     timeseries[i][:] = series[:-1]

        timeseries = [
            series[:-1] for i, series in enumerate(timeseries) if series is not None
        ]

        diff_idx = self.image_starttimes[-1]
        self.ep.replace_ephys(self.ep.ep[:, :diff_idx])

        if "spim" in self.rig:
            self.ep.extract_spim_image_times()
        elif self.rig == "resonant2p":
            self.ep.extract_twophoton_image_times()

        self.compute_imagingtimes_(timeseries=timeseries)

        self.check_align_(timeseries)

        return timeseries  # dask arrays are not immutable, so need for reassignment

    def interp_stacks(self):
        """Interpolate imaging data using stack acquisition onset time

        This was what I started with: assuming each plane of a given
        stack was acquired at the onset of the stack. We, however, do
        have the timing at which each plane was acquired, so this is
        not the right thing to do.

        Use self.interp_im_planes instead.

        """

        from fish.image.vol import InterpArray
        import collections

        self.interp_by_stacks = collections.OrderedDict()
        all_dsets = self.gather_imaging_data()

        print("Interpolating imaging and segmented data between stacks")
        for dset_key, dset in tqdm(all_dsets.items()):

            t_ax = np.where(np.array(dset.shape) == self.nstacks)[0][0]
            self.interp_by_stacks[dset_key] = InterpArray(
                self.ep.stackstart_times, dset, self.ep.im_times.shape, t_ax
            )

    def interp_im_planes(self):
        """Interpolate imaging data using plane acquisition time
        WARNING: obsolete. Use interp_plane_by_plane, a xarray implementation
        """

        import collections
        from fish.image.vol import InterpArray

        plane_start_times = [
            self.ep.planestart_times[plane :: self.num_planes]
            for plane in range(self.num_planes)
        ]

        plane_interp_im = collections.OrderedDict()

        im_dsets = self.gather_im_data()

        print("Interpolating imaging data plane-by-plane")
        for dset_key, dset in tqdm(im_dsets.items()):

            t_ax = np.where(np.array(dset.shape) == self.nstacks)[0][0]
            plane_interp_im[dset_key] = [
                InterpArray(
                    plane_start_times[plane],
                    dset[:, plane, :, :],
                    self.ep.im_times.shape,
                    t_ax,
                )
                for plane in range(self.num_planes)
            ]

        getattr(self, dset_key).interp_im_by_planes = pd.DataFrame(plane_interp_im)

    def interp_plane_by_plane(self, debug=False):
        """Interpolate imaging data using plane acquisition time"""

        import xarray as xr

        plane_start_times = [
            self.ep.planestart_times[plane :: self.num_planes]
            for plane in range(self.num_planes)
        ]

        im_dsets = self.gather_im_data()

        if not np.all(
            np.array(list(map(lambda x: len(x), plane_start_times)))
            == list(im_dsets.values())[0].shape[0]
        ):
            debug = True
            error = True
        else:
            error = False

        if debug:
            fig, ax = plt.subplots(
                len(plane_start_times), 1, figsize=(12, 1.5 * len(plane_start_times))
            )

            for i, plane_start_time in enumerate(plane_start_times):
                ax[i].plot(np.diff(plane_start_time), label=i)
                ax[i].set_title(f"plane {i}")

            fig.tight_layout()

        if not error:
            from microbiome._vendor.analysis_toolbox.utils import rgetattr

            print("Interpolating imaging data plane-by-plane")
            for dset_key, dset in tqdm(im_dsets.items()):
                print(dset_key)

                if self.num_planes != dset.shape[1]:

                    import warnings

                    warnings.warn(
                        f"Mismatch between original numbers of planes {self.num_planes} "
                        f"and number of planes in data {dset.shape[1]}"
                    )

                    plane_start_times = plane_start_times[: dset.shape[1]]

                setattr(
                    rgetattr(self, dset_key.replace(".data", "")),
                    "interp_im_by_planes",
                    [
                        xr.DataArray(
                            dset[:, plane, :, :],
                            dims=("t", "y", "x"),
                            coords={"t": start_times},
                        )
                        for plane, start_times in enumerate(plane_start_times)
                    ],
                )
        else:
            print(f"One of the following t values is not equal")
            print(f"Checking this: {list(im_dsets.values())[0].shape[0]}")
            print("Is equal to all of these:")
            print(np.array(list(map(lambda x: len(x), plane_start_times))))
            print("result")
            print(
                np.array(list(map(lambda x: len(x), plane_start_times)))
                == list(im_dsets.values())[0].shape[0]
            )

    def index_interpolated_planes(self, dset_key, indices):

        return da.stack(
            [
                plane.interp(t=indices).data
                for plane in getattr(getattr(self, dset_key), "interp_im_by_planes")
            ],
            axis=1,
        )

    def interp_seg_planes(self):
        """Interpolate segmented data using plane acquisition time"""

        plane_start_times = [
            self.ep.planestart_times[plane :: self.num_planes]
            for plane in range(self.num_planes)
        ]

        def interp_cells(series):

            from fish.image.vol import InterpArray

            z = series["z"]
            timeseries = series["dff"]
            plane_starts = plane_start_times[z]

            return InterpArray(plane_starts, timeseries, self.ep.im_times.shape, 0)

        for key, value in self.__dict__.items():
            if key.endswith("_seg"):

                print("Interpolating segments between planes")

                value.seg_df["z"] = value.seg_df.progress_apply(value.find_z, axis=1)
                value.seg_df["interp dff"] = value.seg_df.progress_apply(
                    interp_cells, axis=1
                )

    def interp_seg_planebyplane(self, mode="scipy"):
        """Plane-by-plane interpolation of segmented data

        While the interpolation of single cells using plane acquisition
        time doesn't take very long, eventually "indexing" from the
        interpolation cell-by-cell might.

        Instead of working at a cell level, put all cells on the same
        plane into the same dataframe, and put all of those dataframes
        into a larger dataframe. This makes it easier to then work on
        a plane-by-plane basis in a for loop.

        However, this is also not what we want. We want to be able to
        work on a single cell level, not with a for loop, but in
        parallel. The solution to this is to not load the segmented
        data into RAM, but lazily. However, I think this only solves
        part of the problem -- the other part is that the current
        implementation of the interpolation in fish.image.vol doesn't
        work with lazily loaded arrays.
        """

        from fish.image.vol import InterpArray
        from scipy import interpolate
        import xarray as xr

        plane_start_times = [
            self.ep.planestart_times[plane :: self.num_planes]
            for plane in range(self.num_planes)
        ]

        for key, value in self.__dict__.items():

            if key.endswith("_seg") or key.endswith("_voluseg"):

                if "z" not in value.seg_df.columns:
                    value.seg_df["z"] = value.seg_df.progress_apply(
                        value.find_z, axis=1
                    )

                cmpnts_by_plane = []

                print("Interpolating segments bewteen planes, plane-by-plane")
                for plane in tqdm(range(self.num_planes)):

                    cmpnt_plane = {}

                    # find all cells of a certain z plane
                    cmpnts = value.seg_df[value.seg_df["z"] == plane]
                    plane_starts = plane_start_times[plane]

                    cmpnt_plane["index"] = cmpnts.index.to_list()
                    cmpnt_plane["dff"] = np.array(cmpnts["dff"].to_list())

                    if len(cmpnt_plane["index"]) > 0:

                        if mode == "fish":
                            cmpnt_plane["interpolated"] = InterpArray(
                                plane_starts,
                                cmpnt_plane["dff"],
                                self.ep.im_times.shape,
                                1,
                            )
                        elif mode == "xarray":
                            cmpnt_plane["interpolated"] = xr.DataArray(
                                cmpnt_plane["dff"],
                                dims=("neuron", "time"),
                                coords={"time": plane_starts},
                            )
                        elif mode == "scipy":
                            cmpnt_plane["interpolated"] = interpolate.interp1d(
                                plane_starts,
                                cmpnt_plane["dff"],
                                fill_value="extrapolate",
                            )
                    else:
                        cmpnt_plane["interpolated"] = np.nan

                    cmpnts_by_plane.append(cmpnt_plane)

                value.cmpnts_by_plane_df = pd.DataFrame(cmpnts_by_plane)

    def crop_expt(self, start=0, end=None, timeseries=[]):
        """
        Sometimes, there are portions of the experiment that you'd
        not like to analyze. Some example include:

        1. Zebrafish weird behavioural and neural response at the start
           of the experiment.
        2. Shift in the preparation that is not recoverable by registration.

        You might therefore want to remove those parts of the experiment
        before proceeding with analysis.

        Since there needs to be continuity, removing only the start and
        the end of the experiment make sense.

        start and end are both in "imaging time" and in the form of slices,
        where None is equivalent to Matlab's end. Also, since end is the
        "stop" index, it is not inclusive.
        """

        timeseries = [
            series[start:end]
            for i, series in enumerate(timeseries)
            if series is not None
        ]

        start_idx = self.image_starttimes[start]

        if end is None:
            end_idx = end
        else:
            end_idx = self.image_starttimes[end]

        self.ep.replace_ephys(self.ep.ep[:, start_idx:end_idx])
        if self.rig == "resonant2p":
            self.ep.extract_twophoton_image_times()
        elif "spim" in self.rig:
            self.ep.extract_spim_image_times()

        self.compute_imagingtimes_(timeseries=timeseries)

        self.aligned = self.check_align_(timeseries)

        return timeseries  # dask arrays are not immutable, so need for reassignment

    def crop_im_and_ephys(self, start=0, end=None):

        timeseries_keys = list(self.all_dsets.keys())
        timeseries_values = list(self.all_dsets.values())

        results = self.crop_expt(start=start, end=end, timeseries=timeseries_values)

        for key, result in zip(timeseries_keys, results):

            to_execute = f"self.{key} = result"
            print(to_execute)
            exec(to_execute)

    def add_crop_to_metadata_then_crop(self, start=0, end=None):

        self.metadata.add_to_metadata(
            {"crop_expt_start": start, "crop_expt_end": end}, save_and_overwrite=True
        )
        self.metadata_pd = self.metadata.open_metadata(return_df=True)

        self.crop_if_info_in_metadata()

    def crop_if_info_in_metadata(self):

        if "crop_expt_start" in self.metadata_pd.index:

            crop_expt_start = self.metadata_pd.loc["crop_expt_start", "value"]
            crop_expt_end = self.metadata_pd.loc["crop_expt_end", "value"]

            print(
                f"Cropping experiment based on metadata: start {crop_expt_start} end {crop_expt_end}"
            )

            self.crop_im_and_ephys(start=crop_expt_start, end=crop_expt_end)

            return True
        else:
            print(f"crop_expt_start not found in metadata")
            return False

    def kickAssSwimDetect(self):

        from microbiome._vendor.analysis_toolbox.data.ephys import kickAssSwimDetect01

        swimdata = kickAssSwimDetect01(self.ep.leftch, self.ep.rightch)

        self.left_swim = self.ep.leftch
        self.right_swim = self.ep.rightch

        self.left_swimpower = swimdata["fltCh1"]
        self.right_swimpower = swimdata["fltCh2"]
        self.swim_power = (left_swim_power + right_swim_power) / 2

        from microbiome._vendor.analysis_toolbox.preprocess_helper import binarize_swimming

        self.swim_starts = swimdata["swimStartT"]
        self.swim_stops = swimdata["swimEndT"]

        self.swimming, self.swim_start_idx, self.swim_stop_idx = binarize_swimming(
            self.swim_starts, self.swim_stops
        )
        self.swim_power_clipped = self.swim_power * self.swimming

    def detect_swims(
        self,
        leftch_scaling=2.5,
        rightch_scaling=2.5,
        selection="both",
        debug=False,
        from_metadata=True,
        save_to_metadata=False,
    ):
        # overwrite provided channel if channel has been selected before
        if from_metadata:

            if all(
                [
                    attr in self.metadata_pd.index
                    for attr in [
                        "swim_channel",
                        "right_swim_threshold",
                        "left_swim_threshold",
                    ]
                ]
            ):

                selection = self.metadata_pd.loc["swim_channel"]["value"]
                rightch_scaling = self.metadata_pd.loc["right_swim_threshold"]["value"]
                leftch_scaling = self.metadata_pd.loc["left_swim_threshold"]["value"]

                print("Saved swim channel found and set!")
                print("----------------------------------")
                print(f"Channel: {selection}")
                print(f"Right swim threshold: {rightch_scaling}")
                print(f"Left swim threshold: {leftch_scaling}")

            elif all(
                [
                    attr in self.metadata_pd.index
                    for attr in [
                        "swim_channel",
                        "swim_threshold",
                    ]
                ]
            ):

                selection = self.metadata_pd.loc["swim_channel"]["value"]
                selection = selection.replace("ch", "")

                if selection == "right" or selection == "both":
                    rightch_scaling = self.metadata_pd.loc["swim_threshold"]["value"]
                if selection == "left" or selection == "both":
                    leftch_scaling = self.metadata_pd.loc["swim_threshold"]["value"]

                print("Old saved swim channel found and set!")
                print("----------------------------------")
                print(f"Channel: {selection}")
                print(f"Right swim threshold: {rightch_scaling}")
                print(f"Left swim threshold: {leftch_scaling}")

            else:
                print("Saved swim channel not found...")
                print("Using provided values...")

        else:
            if save_to_metadata:

                # add information to metadata file
                add = {
                    "swim_channel": selection,
                    "left_swim_threshold": leftch_scaling,
                    "right_swim_threshold": rightch_scaling,
                }

                self.metadata.add_to_metadata(add, save_and_overwrite=True)
                print("Channel selected saved!")

        (
            left_swim_power,
            left_swim_power_clipped,
            left_swim_starts,
            left_swim_stops,
            left_swimstart_idx,
            left_swimstop_idx,
            left_swimming,
        ) = self.ep.find_swims(self.ep.leftch, scaling=leftch_scaling)

        (
            right_swim_power,
            right_swim_power_clipped,
            right_swim_starts,
            right_swim_stops,
            right_swimstart_idx,
            right_swimstop_idx,
            right_swimming,
        ) = self.ep.find_swims(self.ep.rightch, scaling=rightch_scaling)

        self.left_swim = self.ep.leftch
        self.right_swim = self.ep.rightch

        self.left_swimpower = left_swim_power
        self.right_swimpower = right_swim_power

        if selection == "both":
            self.swim_power = (left_swim_power + right_swim_power) / 2
        elif selection == "left" or selection == "leftch":
            self.swim_power = left_swim_power
        elif selection == "right" or selection == "rightch":
            self.swim_power = right_swim_power

        self.left_swimpower_clipped = left_swim_power_clipped
        self.right_swimpower_clipped = right_swim_power_clipped

        if selection == "both":
            self.swim_power_clipped = (
                left_swim_power_clipped + right_swim_power_clipped
            ) / 2
        elif selection == "left" or selection == "leftch":
            self.swim_power_clipped = self.left_swimpower_clipped
        elif selection == "right" or selection == "rightch":
            self.swim_power_clipped = self.right_swimpower_clipped

        self.left_swimstarts = left_swim_starts
        self.right_swimstarts = right_swim_starts
        # self.swim_starts = np.logical_or(left_swim_starts, right_swim_starts)

        self.left_swimstops = left_swim_stops
        self.right_swimstops = right_swim_starts
        # self.swim_stops = np.logical_or(left_swim_stops, right_swim_stops)

        self.left_swimstart_idx = left_swimstart_idx
        self.right_swimstart_idx = right_swimstart_idx
        # self.swim_start_idx = np.sort(
        #     np.concatenate([self.left_swimstart_idx, self.right_swimstart_idx])
        # )

        self.left_swimstop_idx = left_swimstop_idx
        self.right_swimstop_idx = right_swimstop_idx
        # self.swim_stop_idx = np.sort(
        #     np.concatenate([self.left_swimstop_idx, self.right_swimstop_idx])
        # )

        self.left_swimming = left_swimming
        self.right_swimming = right_swimming

        if selection == "both":
            self.swimming = np.logical_or(left_swimming, right_swimming)
        elif selection == "left" or selection == "leftch":
            self.swimming = left_swimming
        elif selection == "right" or selection == "rightch":
            self.swimming = right_swimming

        from microbiome._vendor.analysis_toolbox.math_helper import find_change

        self.swim_starts = find_change(
            self.swimming, np.nanmin(self.swimming), np.nanmax(self.swimming)
        )
        self.swim_stops = find_change(
            self.swimming, np.nanmax(self.swimming), np.nanmin(self.swimming)
        )
        self.swim_start_idx = np.where(self.swim_starts)[0]
        self.swim_stop_idx = np.where(self.swim_stops)[0]

        if selection == "both":
            self.swimming_bias = left_swimming.astype("int") + right_swimming.astype(
                "int"
            )
            self.swimming_forward = self.swimming_bias == 2

        if debug:
            self.visualize_swim_detection()

        print("Done!\n")

    def visualize_swim_detection(self):

        fig, ax = plt.subplots(4, 1, figsize=(12, 6), sharex=True)

        from microbiome._vendor.analysis_toolbox.utils import center_minmax

        ax[0].plot(self.left_swimpower)
        ax[0].plot(-self.right_swimpower)

        ymin, ymax = center_minmax(
            np.concatenate([self.left_swimpower, -self.right_swimpower]), 0.01, 99.9
        )
        ax[0].set_ylim([ymin, ymax])

        ax[1].plot(self.left_swimpower_clipped)
        ax[1].plot(-self.right_swimpower_clipped)
        ax[1].set_ylim([ymin, ymax])

        ax[2].plot(self.swim_power_clipped)
        ax[2].set_ylim([0, ymax])

        ax[3].plot(self.swimming)

        return fig, ax

    def set_swims(
        self, channel, scaling=1.6, from_metadata=False, save_to_metadata=False
    ):

        selected_channel = None  # string description of channel
        selected_channel_scaling = scaling

        # overwrite provided channel if channel has been selected before
        if from_metadata:

            try:
                self.metadata_pd.loc[["swim_channel", "swim_threshold"]]
                selected_channel = self.metadata_pd.loc["swim_channel"]["value"]
                selected_channel_scaling = self.metadata_pd.loc["swim_threshold"][
                    "value"
                ]

                if selected_channel == "both":
                    channel = (self.ep.leftch + self.ep.rightch) / 2
                else:
                    channel = eval("self.ep." + selected_channel)

                print("Saved swim channel found and set!")
                print("----------------------------------")
                print(f"Channel: {selected_channel}")
                print(f"Threshold: {selected_channel_scaling}")

            except:
                print("Saved swim channel not found...")

        else:

            # Channel can either be a numpy array of the swim ephys
            # channel or a string indicating which channel to pick
            # from this dataset.
            if isinstance(channel, str):

                selected_channel = channel
                selected_channel_scaling = scaling

                if save_to_metadata:

                    # add information to metadata file
                    add = {"swim_channel": channel, "swim_threshold": scaling}

                    self.metadata.add_to_metadata(add, save_and_overwrite=True)
                    print("Channel selected saved!")

                if selected_channel == "both":
                    channel = (self.ep.leftch + self.ep.rightch) / 2
                else:
                    print("Finding channel data...")
                    channel = eval("self.ep." + selected_channel)

            else:
                print(
                    """Channel information not added to metadata. To
                      add to metadata file, use a string channel input
                      instead of the swim channel itself."""
                )

        from fish.ephys.ephys import load, windowed_variance, estimate_swims

        self.swim = channel

        if selected_channel == "both":
            self.swim_power = (
                windowed_variance(self.ep.leftch)[0]
                + windowed_variance(self.ep.rightch)[0]
            ) / 2
        else:
            self.swim_power = windowed_variance(channel)[0]

        self.swim_starts, self.swim_stops, _ = estimate_swims(
            self.swim_power, scaling=selected_channel_scaling
        )

        from microbiome._vendor.analysis_toolbox.preprocess_helper import binarize_swimming

        self.swimming, self.swim_start_idx, self.swim_stop_idx = binarize_swimming(
            self.swim_starts, self.swim_stops
        )
        self.swim_power_clipped = self.swim_power * self.swimming

    def compute_max_imintvl(self):
        """
        PIPELINE (6)
        Compute maximum imaging intervals so that we can use this
        information to bin our ephys data to match the number of
        images.

        show_stats: always show stats so that we can determine if
                    e.g. it makes any sense at all to use self.im_rate
        """

        max_interval = np.max(self.im_intervals)

        self.downsample_factor = max_interval

        if self.debug:

            from microbiome._vendor.analysis_toolbox.utils import unique

            print("Imaging intervals: %s" % str(unique(self.im_intervals)))
            print("Bins of Imaging intervals: %s" % np.histogram(self.im_intervals)[0])
            print("Imaging histo counts: %s" % np.histogram(self.im_intervals)[1])
            print("Largest interval: %i" % max_interval)

            imtint_fig, imtint_ax = plt.subplots(
                2,
                1,
                figsize=(9, 6),
                facecolor="w",
                gridspec_kw={"height_ratios": [1, 3]},
            )
            imtint_ax[0].plot(
                self.image_starttimes[0:-1] / self.ephys_rate, self.im_intervals
            )
            imtint_ax[0].set_ylabel("Intervals")
            imtint_ax[0].set_xlabel("Time (s)")
            imtint_ax[0].set_xlim([0, 10])

            imtint_ax[1].hist(self.im_intervals, align="left")
            imtint_ax[1].set_yscale("log", nonposy="clip")
            imtint_ax[1].set_ylabel("log(count)")
            imtint_ax[1].set_xlabel("Intervals")

            imtint_fig.tight_layout()

    def chop_ephys(self, timeseries):
        """Chop up ephys trace according to when the image was acquired.

        For stacks, each stack is considered to be a single time point.
        For single planes, each plane is considered to be a single time
        point.
        """

        chopped_series = np.empty([self.nstacks, self.downsample_factor])
        chopped_series.fill(np.nan)

        final_frame = self.ep.ep.shape[1] - 1

        for i in range(
            0, self.nstacks
        ):  # range from 0 to nstack-1 (ie excluding nstacks)

            start = self.image_starttimes[i]

            if i == len(self.image_starttimes) - 1:  # if last frame

                if final_frame - start < self.downsample_factor:
                    stop = final_frame
                else:
                    stop = start + self.downsample_factor

            else:
                stop = self.image_starttimes[i + 1]

            chopped_series[i, : stop - start] = timeseries[start:stop]

        return chopped_series

    def verify_binning(self, timeseries, binned):

        fig, ax = plt.subplots(2, 1, figsize=(9.5, 4), sharex=True)
        ax[0].plot(
            np.linspace(0, len(timeseries) / self.ephys_rate, num=len(timeseries)),
            timeseries,
            color="tab:blue",
            label="timeseries",
        )
        ax[0].plot(
            self.image_starttimes / self.ephys_rate,
            binned,
            color="tab:orange",
            label="binned",
        )
        ax[0].legend()

        ax[1].plot(
            self.image_starttimes / self.ephys_rate,
            binned,
            color="tab:orange",
            label="binned",
        )

        for start_times in self.image_starttimes:
            ax[1].axvline(x=start_times / self.ephys_rate, color="k", linewidth=0.5)

        ax[-1].set_xlabel("Time [s]")
        fig.tight_layout()

    def bin_ephys(self, timeseries, nanfunc, axis=1, debug=False):
        """Chop ephys according to image acquisition times and then
        bin each segment.

        Must run self.compute_max_imintvl first.
        """

        if self.downsample_factor is None:
            self.compute_max_imintvl()

        from microbiome._vendor.analysis_toolbox.utils import nanfuncs

        func = nanfuncs[nanfunc]

        chopped = self.chop_ephys(timeseries)

        if nanfunc == "nanmode":
            binned = np.squeeze(func(chopped, axis=axis))[0]
        else:
            binned = func(chopped, axis=axis)

        if self.debug or debug:
            self.verify_binning(timeseries, binned)

        return chopped, binned

    def downsample_ephys(self, swimming=False, ephys_timeseries={}, debug=False):

        if swimming:
            self.downsample_swimming()

        from microbiome._vendor.analysis_toolbox.utils import setattrs

        for ephys_name, bin_method in ephys_timeseries.items():
            chopped, ds = self.bin_ephys(
                getattr(self.ep, ephys_name), bin_method, debug=debug
            )
            setattr(self, f"chopped_{ephys_name}", chopped)
            setattr(self, f"ds_{ephys_name}", ds)

    def downsample_ephys_vars(
        self,
        ephys_timeseries={
            "stimulus_brightness": "nanmean",
            "gain": "nanmean",
            "current_epoch": "nanmax",
            "trial_num": "nanmax",
            "command_stimulus_velocity": "nanmax",
            "observed_stimulus_velocity": "nanmax",
        },
        debug=False,
    ):
        # Discretize swims and find frames with swim starts and stops
        # downsample swim power
        swimming = self.swim_power is not None

        if not swimming:
            print("Swim trace not set!")

        self.downsample_ephys(swimming=swimming, ephys_timeseries=ephys_timeseries)

        if (self.debug or debug) and swimming:

            self.dataset.verify_binning(
                self.dataset.swim_power, self.dataset.ds_swimpower
            )

    def divide_into_trials(self, additional_vars_dict={}):
        """
        additional_vars_dict [dict]:
        possible: self.ep.stimulus_id, self.ep.optogenetics_trigger, self.im_CM0_voluseg.cell_dff
        """

        self.downsample_ephys_vars()

        # Put ephys data into dictionary
        default_vars_dict_ephysrate = {
            "time": self.ep.t_s,
            "leftch": self.ep.leftch,
            "rightch": self.ep.rightch,
            "swimpower": self.swim_power,  # channel 1+2
            "swimpower_clipped": self.swim_power_clipped,
            "swimming": self.swimming,  # channel 1+2
            "swimstarts": self.swim_starts,  # channel 1+2
            "swimstops": self.swim_stops,  # channel 1+2
            "camtrig": self.ep.im_times,  # channel 3
            "epoch": self.ep.current_epoch,
            "cmdvel": self.ep.command_stimulus_velocity,
            "brightness": self.ep.stimulus_brightness,
            "gain": self.ep.gain,
            "obsvel": self.ep.observed_stimulus_velocity,
        }

        # ephys data downsampled to imaging rate
        default_vars_dict_imrate = {
            # missing raw channel 1 and 2
            "ds_swimpower": self.ds_swimpower,
            "ds_swimpower_clipped": self.ds_swimpower_clipped,
            "ds_epoch": self.ds_current_epoch,
            "ds_swimming": self.ds_swimming,
            "ds_swimstarts": self.ds_swimstarts,
            "ds_swimstops": self.ds_swimstops,
            "ds_cmdvel": self.ds_command_stimulus_velocity,
            "ds_brightness": self.ds_stimulus_brightness,
            "ds_obsvel": self.ds_observed_stimulus_velocity,
            "ds_gain": self.ds_gain,
        }

        # imaging data
        default_im_dict_imrate = dict()
        if hasattr(self, "im_dff"):
            default_im_dict_imrate["im_dff"] = self.im_dff.data
        if hasattr(self, "im_ds"):
            default_im_dict_imrate["im_ds"] = self.im_ds.data
        if hasattr(self, "im_raw"):
            default_im_dict_imrate["im_raw"] = self.im_raw.data
        if hasattr(self, "im_voluseg"):
            default_im_dict_imrate["cmpnt_dff"] = self.seg.cmpnt_dff.transpose()

        from microbiome._vendor.analysis_toolbox.trial_helper import divide_into_trials

        self.vars_D_ephysrate = divide_into_trials(
            self.ep.trial_num, default_vars_dict_ephysrate, trial_inds_name="trial_num"
        )
        self.vars_D_imrate = divide_into_trials(
            self.ds_trial_num, default_vars_dict_imrate, trial_inds_name="ds_trial_num"
        )

        # Merge dictionaries
        self.spim_dict = {
            **self.vars_D_ephysrate,
            **self.vars_D_imrate,
        }

        return pd.DataFrame(self.spim_dict)

    def downsample_swimming(self):
        """
        It is difficult to decide how to bin the various behavioral
        variables. It also depends if the images are frame-triggered
        properly. But given that the images are indeed frame-triggered
        properly (there's no reason why they shouldn't), then I think
        there are some rules for binning them.

        For analog signals like ventral root recording (voltage) and
        swim power (power of voltage), it makes the most sense to use
        'max', since calcium dynamics are typically integrative (and
        sometimes even non-linearly so). However, other times, I think
        'mean' makes much more sense.

        For boolean signals like whether or not the fish is swimming,
        I think mode makes the most sense. For things like epoch too,
        where the signal is more digital and discrete, the mode will
        also prevent things like intermediate values that is difficult
        to interpret.
        """

        if self.downsample_factor is None:
            self.compute_max_imintvl()

        # self.swim is not set when both ephys channels are used (e.g. detect_swims())
        if self.swim is not None:
            self.chopped_swim, self.ds_swim = self.bin_ephys(
                self.swim, nanfunc="nanmax"
            )
        self.chopped_swim_power, self.ds_swimpower = self.bin_ephys(
            self.swim_power, nanfunc="nanmax"
        )
        self.chopped_swim_power_clipped, self.ds_swimpower_clipped = self.bin_ephys(
            self.swim_power_clipped, nanfunc="nanmax"
        )
        self.chopped_swimming, self.ds_swimming = self.bin_ephys(
            self.swimming, nanfunc="nanmax"
        )

        self.chopped_swim_starts = self.chop_ephys(self.swim_starts)
        self.chopped_swim_starts[np.isnan(self.chopped_swim_starts)] = 0
        self.chopped_swim_stops = self.chop_ephys(self.swim_stops)
        self.chopped_swim_stops[np.isnan(self.chopped_swim_stops)] = 0

        self.ds_swimstarts = np.nansum(self.chopped_swim_starts, axis=1)
        self.ds_swimstops = np.nansum(self.chopped_swim_stops, axis=1)

        ## TODO: I don't know if this is the best way to derived the downsampled
        ## swim starts and stops. Might be better to work with self.swim_start_idx
        ## so that the number of swims are the same.
        self.ds_swim_start_idx = self.index_onsets(self.chopped_swim_starts)
        self.ds_swim_stop_idx = self.index_onsets(self.chopped_swim_stops)

        self.check_and_correct_different_timing_lengths()

        if not self.check_mismatched_timings(
            self.ds_swim_start_idx, self.ds_swim_stop_idx
        ):
            self.swimstart_idx, self.swimstop_idx = self.correct_mismatched_timings()

    def check_and_correct_different_timing_lengths(self):

        # check whether the only discrepancy is length
        for i, (start_idx, stop_idx) in enumerate(
            zip(self.ds_swim_start_idx, self.ds_swim_stop_idx)
        ):
            if start_idx > stop_idx:
                print(i, start_idx, stop_idx)
                break

        arr_names = ["ds_swim_start_idx", "ds_swim_stop_idx"]
        arrs = [getattr(self, arr_name) for arr_name in arr_names]
        arr_lens = np.array([len(arr) for arr in arrs])
        print(f"ds_swim_start_idx, ds_swim_stop_idx lens: {arr_lens}")
        shorter_arr_len = np.nanmin(arr_lens)
        longer_arr_len = np.nanmax(arr_lens)

        if shorter_arr_len == longer_arr_len:

            return None

        elif i == shorter_arr_len - 1:
            print(f"No discrepancies, probably just missing an index")

            if longer_arr_len - shorter_arr_len == 1:

                shorter_arr_idx = int(np.argwhere(arr_lens == shorter_arr_len))
                longer_arr_idx = int(np.argwhere(arr_lens == longer_arr_len))
                new_shorter_arr = arrs[longer_arr_idx]
                new_shorter_arr[: len(arrs[shorter_arr_idx])] = arrs[shorter_arr_idx]

                print(f"{arr_names[shorter_arr_idx]} will be modified!")
                setattr(self, arr_names[shorter_arr_idx], new_shorter_arr)

            else:
                raise ValueError("Can't understand swim starts and stops")
        else:
            raise ValueError("Can't understand swim starts and stops")

    def check_mismatched_timings(self, swimstart_index, swimstop_index):

        if (swimstop_index > swimstart_index).sum() > 0:

            warnings.warn(
                "Some swim stops happen before their \
            corresponding swim starts. You might want to check..."
            )

            return False

        else:

            return True

    def correct_mismatched_timings(self):

        swimstart_list = list(self.ds_swim_start_idx)
        swimstop_list = list(self.ds_swim_stop_idx)

        if len(swimstart_list) - len(swimstop_list) != 0:

            from microbiome._vendor.analysis_toolbox.utils import correct_mismatched_starts_and_stops

            swimstart_list, swimstop_list = correct_mismatched_starts_and_stops(
                swimstart_list, swimstop_list
            )
            self.check_mismatched_timings(
                np.array(swimstart_list), np.array(swimstop_list)
            )

        else:
            print("Same number of starts and stops!")

        return np.array(swimstart_list), np.array(swimstop_list)

    def index_onsets(self, chopped):

        chopped_index = []

        for i, row in enumerate(chopped):
            n = int(row.sum())

            for event in range(n):
                chopped_index.append(i)

        assert np.sum(chopped) == len(chopped_index)

        return np.array(chopped_index)

    def check_frame_trigger(self, event_trace, trigger=(0, 1)):
        """
        Create a plot with camera trigger and epoch triggered on some
        event.
        """

        from microbiome._vendor.analysis_toolbox.math_helper import find_change

        events = find_change(event_trace, trigger[0], trigger[1])
        event_idx = np.where(events)[0]

        print(f"Number of events found: {len(event_idx)}")

        before_trig = 0.5
        after_trig = 5
        window = np.arange(
            np.floor(-before_trig * self.ephys_rate),
            np.ceil(after_trig * self.ephys_rate) + 1,
        )

        trig_window = event_idx[:, np.newaxis] + window

        incl_trig = np.logical_not(
            np.any(trig_window > len(self.ep.im_times) - 1, axis=1)
        )
        trig_window = trig_window[incl_trig].astype("int")

        trig_imtimes = self.ep.im_times[trig_window]
        trig_cmdvel = self.ep.channel9[trig_window]
        trig_epoch = self.ep.channel6[trig_window]
        trig_event = event_trace[trig_window]

        ft_fig, ft_ax = plt.subplots(4, 1, figsize=(8.5, 7), sharex=True)

        ft_ax[0].plot(trig_imtimes.T)
        ft_ax[0].set_ylabel("Camera trigger")

        ft_ax[1].plot(trig_cmdvel.T)
        ft_ax[1].set_ylabel("Command velocity")

        ft_ax[2].plot(trig_epoch.T)
        ft_ax[2].set_ylabel("Epoch")

        ft_ax[3].plot(trig_event.T)
        ft_ax[3].set_ylabel("Event")

        ft_ax[-1].set_xlabel("Time [frame]")

        ft_fig.tight_layout()
        ft_ax[0].set_xlim([2500, 3500])

    def pad_swimming(self, pre_swim, post_swim):

        before_pad = np.unique(
            np.ravel(
                self.swim_start_idx[np.newaxis, :].transpose()
                + np.arange(-pre_swim * self.ephys_rate, 0)
            )
        )
        before_pad = before_pad[np.logical_not(before_pad < 0)]

        after_pad = np.unique(
            np.ravel(
                self.swim_stop_idx[np.newaxis, :].transpose()
                + np.arange(0, post_swim * self.ephys_rate)
            )
        )
        after_pad = before_pad[np.logical_not(before_pad > len(self.swimming))]

        swimming_pad = self.swimming.copy()
        swimming_pad[before_pad] = 1
        swimming_pad[after_pad] = 1

        return swimming_pad

    def construct_trig_window_ep(self, before_trig, after_trig):

        from microbiome._vendor.analysis_toolbox.trial_helper import construct_trig_window

        return construct_trig_window(before_trig, after_trig, self.ephys_rate)

    def construct_trig_window_im_from_window_ep(self, before_trig, after_trig):

        t_axis_ep, window_ep = self.construct_trig_window_ep(before_trig, after_trig)

        window_im = self.sample_trig_window_im(window_ep)
        t_axis_im = window_im / self.ephys_rate

        return t_axis_im, window_im

    def sample_trig_window_im(self, window_ep):
        """
        window_ep must contain 0
        """
        from microbiome._vendor.analysis_toolbox.utils import sample_indices

        window_im = sample_indices(window_ep, self.ephys_rate, self.im_rate, align=0)

        return window_im

    def event_triggered(self, window, event_ind):

        from microbiome._vendor.analysis_toolbox.trial_helper import event_triggered

        return event_triggered(window, event_ind, self.ep.rightch.shape[0])

    def generate_triggered_indices(self, before_trig, after_trig, event_ind, mode="ep"):

        if mode == "ep":
            t_axis, window = self.construct_trig_window_ep(before_trig, after_trig)
        elif mode == "im":
            t_axis, window = self.construct_trig_window_im_from_window_ep(
                before_trig, after_trig
            )

        return self.event_triggered(window, event_ind)

    def generate_filtered_triggered_indices(
        self, before_trig, after_trig, event_ind, mode="ep", filters=[]
    ):

        unfiltered_trig_inds, valid = self.generate_triggered_indices(
            before_trig, after_trig, event_ind, mode=mode
        )
        print(unfiltered_trig_inds.shape)

        if filters:
            filter_list = []

            for filt in filters:
                # (1) everything must be True
                # (2) cut off excess
                filt_bool = np.any(
                    self.generate_triggered(
                        filt["extract_from"],
                        filt["before_trig"],
                        filt["after_trig"],
                        event_ind,
                        match_valid=valid,
                    ),
                    axis=1,
                )
                if filt["not"]:
                    filt_bool = np.logical_not(filt_bool)
                filter_list.append(filt_bool)

            agg_filter = np.logical_and.reduce(filter_list)

            return unfiltered_trig_inds[agg_filter]
        else:
            return unfiltered_trig_inds

    def generate_triggered(
        self,
        extract_from,
        before_trig,
        after_trig,
        event_ind,
        mode="ep",
        match_valid=None,
    ):
        """Generate a triggered array

        OUTPUT
        =======
        triggered [np.ndarray]: trials x time
        """

        triggered_inds, valid = self.generate_triggered_indices(
            before_trig, after_trig, event_ind, mode=mode
        )

        if match_valid is not None:
            assert len(valid) >= len(match_valid)

            # remove extra False's in match_valid
            # (1) Find out which of the current triggered_inds should be Falses
            # (2) Remove them
            triggered_inds = triggered_inds[match_valid[valid]]

        return extract_from[triggered_inds]

    def setup_for_trial_processing(self):

        self.vars_dict_ephysrate = {
            "time": self.ep.t_s,
            "leftch": self.ep.leftch,
            "rightch": self.ep.rightch,
            "swim": self.swim,
            "swimpower": self.swim_power,  # channel 1+2
            "swimming": self.swimming,  # channel 1+2
            "swimstarts": self.swim_starts,  # channel 1+2
            "swimstops": self.swim_stops,  # channel 1+2
            "camtrig": self.ep.im_times,  # channel 3
            "epoch": self.ep.channel6,
            "cmdvel": self.ep.channel9,
            "brightness": self.ep.channel8,
            "gain": self.ep.channel5,
            "obsvel": self.ep.channel4,
        }

    def generate_filled_volume(
        self, values, im_mode, valid_index=None, volume=None, **kwargs
    ):

        subobjects = im_mode.split(".")

        if (
            "_seg" in im_mode
            or "_voluseg" in im_mode
            or im_mode == "seg_extras"
            or "regd" in im_mode
        ):

            if valid_index is None:
                valid_index = np.arange(getattr(self, subobjects[0]).n)

            if im_mode == "seg_extras" or "regd" in im_mode:
                vol = rgetattr(self, im_mode).find_cells(
                    valid_index, values, cell_volume=volume, **kwargs
                )

            elif "_voluseg" in im_mode:
                vol = rgetattr(self, im_mode).find_cells(
                    valid_index, mask=values, cell_volume=volume, **kwargs
                )

            elif "_seg" in im_mode:
                vol = rgetattr(self, im_mode).find_cells(
                    valid_index, cell_volume=volume, mask=values, **kwargs
                )
        elif (
            im_mode.endswith("_dff")
            or im_mode.endswith("_cis")
            or im_mode.endswith("_cit")
            or im_mode.endswith("raw")
        ):

            if valid_index is not None:

                ## the indices are linearised and will have to be eventually reshaped
                if isinstance(valid_index, np.ndarray):

                    if volume is None:
                        voxels = np.full(getattr(self, im_mode).nvoxels, np.nan)
                    else:
                        voxels = volume.flatten()

                    voxels[valid_index] = values
                    values = voxels

                    vol = np.reshape(
                        values.to_numpy(), getattr(self, im_mode).stack_shape
                    )

                elif isinstance(valid_index, list) and len(valid_index) == 3:

                    if volume is None:

                        vol = np.full(getattr(self, im_mode).stack_shape, np.nan)

                    else:

                        vol = volume

                    vol[*valid_index] = values

        return vol

    def generate_layered_volume(
        self, im_mode, labels, cell_labels, intersect=None, weight_multiplier=1
    ):
        """Generate a layered volume using labels provided"""

        volume = np.full(getattr(self, im_mode).volume_mean.shape, np.nan)

        ## iterate over every single cluster
        for ilabel, label in enumerate(tqdm(labels)):

            if len(label) > 0:

                if cell_labels.ndim == 2:
                    from microbiome._vendor.analysis_toolbox.data.segments import return_cells_in_region

                    cells_id = return_cells_in_region(cell_labels, label)
                else:
                    cells_id = np.concatenate(
                        [np.where(cell_labels == lab)[0] for lab in label]
                    )

                if intersect is not None:
                    cells_id = np.intersect1d(cells_id, intersect)

                ## Plot the spatial location of all the cells that are found within that ROI
                volume = getattr(self, im_mode).find_cells(
                    cells_id,
                    cell_volume=volume,
                    mask="weights",
                    weight_multiplier=weight_multiplier,
                )

        return volume

    def plot_layered_clusters_in_brain(
        self, im_mode, labels, cell_labels, intersect=None, debug=False
    ):

        alpha_volume = self.generate_layered_volume(
            im_mode, labels, cell_labels, intersect=intersect
        )

        if debug:

            from microbiome._vendor.analysis_toolbox.stacks_helper import make_projections
            from microbiome._vendor.analysis_toolbox.utils import now_str

            alpha_proj = self.make_projections(
                alpha_volume,
                im_mode,
                cmap="viridis",
                perspective=False,
                max_p=True,
                suptitle=f"Interesting clusters",
                xlabel="",
                savename=self.savepath + f"alpha-cells_" + now_str() + ".png",
            )

        vols = self.plot_clusters_in_brain(
            im_mode,
            cell_labels,
            cell_indices=np.arange(len(cell_labels)),
            included_cell_indices=intersect,
            labels_to_plot=labels,
            individually=True,
            together=True,
            make_plots=debug,
        )

        from microbiome._vendor.analysis_toolbox.stacks_helper import make_projections

        colors, cmap, proj_fig, proj_ax, proj_gs = self.make_projections(
            vols["all"],
            im_mode,
            cmap="rainbow",
            alpha=alpha_volume,
            perspective=True,
            max_p=False,
            vmin=0,
            vmax=len(labels) - 1,
            compact=True,
            suptitle=f"Stimulus-triggered clusters",
            xlabel="Cluster #",
            savename=self.savepath
            + f"stim-trig-clusts_full-proj_"
            + now_str()
            + ".svg",
        )

        slabs_alpha = self.make_slabs(
            vols["all"],
            im_mode,
            num_slabs=6,
            alpha=alpha_volume,
            cmap="rainbow",
            perspective=True,
            max_p=False,
            vmin=0,
            vmax=len(labels) - 1,
            suptitle=f"Stimulus-triggered clusters",
            xlabel="Cluster #",
            savename=self.savepath
            + f"stim-trig-clusts_slab-proj_"
            + now_str()
            + ".svg",
        )

        return colors, cmap, proj_fig, proj_ax, proj_gs

    def plot_clusters_in_brain(
        self,
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
        **generate_filled_volume_kwargs,
    ):

        from microbiome._vendor.analysis_toolbox.cell_helper import plot_clusters_in_brain

        return plot_clusters_in_brain(
            self,
            im_mode,
            labels,
            cell_indices,
            included_cell_indices=included_cell_indices,
            individually=individually,
            individual_labels=individual_labels,
            together=together,
            labels_to_plot=labels_to_plot,
            z_offset=z_offset,
            y_offset=y_offset,
            x_offset=x_offset,
            cmap=cmap,
            make_plots=make_plots,
            **generate_filled_volume_kwargs,
        )

    def make_projections(
        self,
        image_stack,
        im_mode,
        image_slice=(slice(None), slice(None), slice(None)),
        background=False,
        **kwargs,
    ):

        from microbiome._vendor.analysis_toolbox.stacks_helper import make_projections

        if not hasattr(rgetattr(self, im_mode), "vox_spacing"):
            rgetattr(self, im_mode).compute_vox_spacing()

        if background is not None:

            if isinstance(background, bool):
                background = (
                    rgetattr(self, im_mode).volume_mean[image_slice].astype("float32")
                )
                background_vmin = np.nanpercentile(
                    rgetattr(self, im_mode).volume_mean[image_slice], 10
                )
                background_vmax = np.nanmax(
                    rgetattr(self, im_mode).volume_mean[image_slice]
                )
            else:
                print("Using provided background...")
                background = background[image_slice].astype("float32")
                background_vmin = np.nanpercentile(background, 0)
                background_vmax = np.nanmax(background[image_slice])

        else:
            background = None
            background_vmin = 0
            background_vmax = 100

        if self.rig == "nikon2p":
            print("Reversing in z...")
            image_stack = image_stack[::-1]
            if background is not None:
                background = background[::-1]

        return make_projections(
            image_stack[image_slice],
            background=background,
            background_vmode="raw",
            background_vmin=background_vmin,
            background_vmax=background_vmax,
            voxel_spacing=rgetattr(self, im_mode).vox_spacing,
            rotate180=self.rig == "spim2",
            **kwargs,
        )

    def make_slabs(self, image_stack, im_mode, **kwargs):

        from microbiome._vendor.analysis_toolbox.stacks_helper import make_slabs

        return make_slabs(
            image_stack,
            background=getattr(self, im_mode).volume_mean.astype("float32"),
            background_vmode="raw",
            background_vmin=np.nanpercentile(getattr(self, im_mode).volume_mean, 10),
            background_vmax=np.nanmax(getattr(self, im_mode).volume_mean),
            rotate180=self.rig == "spim2",
            **kwargs,
        )

    def create_animation(self, tyx_vols, im_mode, **kwargs):

        from microbiome._vendor.analysis_toolbox.movie_helper import create_animation

        return create_animation(
            tyx_vols,
            rotate180=self.rig == "spim2",
            voxel_spacing=getattr(self, im_mode).vox_spacing,
            **kwargs,
        )

    def swim_as_trials(self):

        self.total_swim_power_per_bout = []
        self.mean_swim_power_per_bout = []
        self.max_swim_power_per_bout = []

        for swim_start_idx, swim_stop_idx in zip(
            self.swim_start_idx, self.swim_stop_idx
        ):

            self.total_swim_power_per_bout.append(
                np.nansum(self.swim_power[swim_start_idx:swim_stop_idx])
            )

            self.mean_swim_power_per_bout.append(
                np.nanmean(self.swim_power[swim_start_idx:swim_stop_idx])
            )

            self.max_swim_power_per_bout.append(
                np.nanmax(self.swim_power[swim_start_idx:swim_stop_idx])
            )

        self.swim_intervals = self.swim_start_idx[1:] - self.swim_stop_idx[:-1]

        self.gain_at_start = self.ep.gain[self.swim_start_idx[:-1]]
        self.vel_at_start = self.ep.command_stimulus_velocity[self.swim_start_idx[:-1]]
        self.epoch_at_start = self.ep.current_epoch[self.swim_start_idx[:-1]]

        self.gain_at_stop = self.ep.gain[self.swim_stop_idx[:-1]]
        self.vel_at_stop = self.ep.command_stimulus_velocity[self.swim_stop_idx[:-1]]
        self.epoch_at_stop = self.ep.current_epoch[self.swim_stop_idx[:-1]]

        self.swim_intervals_df = pd.DataFrame(
            {
                "interval": self.swim_intervals,
                "total_power": self.total_swim_power_per_bout[:-1],
                "mean_power": self.mean_swim_power_per_bout[:-1],
                "max_power": self.max_swim_power_per_bout[:-1],
                "start": self.swim_start_idx[:-1],
                "stop": self.swim_stop_idx[:-1],
                "restart": self.swim_start_idx[1:] - 1,
                "stop_gain": self.gain_at_stop,
                "stop_vel": self.vel_at_stop,
                "stop_epoch": self.epoch_at_stop,
                "start_gain": self.gain_at_start,
                "start_vel": self.vel_at_start,
                "start_epoch": self.epoch_at_start,
            }
        )

        def generate_trial_inds(trial):

            trial_start = trial["start"]
            trial_stop = trial["restart"]

            return np.arange(trial_start, trial_stop + 1).astype("int")

        self.swim_intervals_df["trial_inds_ep"] = self.swim_intervals_df.progress_apply(
            generate_trial_inds, axis=1
        )

        display(self.swim_intervals_df.head())

    def compute_volume_mean(self, debug=False):

        from distributed import Client

        if "im_CM0_dff" in self.im_modes:

            if not hasattr(self.im_CM0_dff, "volume_mean"):

                if "im_CM0_ds" in self.im_modes:

                    self.im_CM0_dff.volume_mean = da.nanmean(
                        self.im_CM0_ds.data[
                            self.im_CM0_ds.data.shape[0] // 2
                            - 25 : self.im_CM0_ds.data.shape[0] // 2
                            + 25
                        ],
                        axis=0,
                    )

                    display(self.im_CM0_dff.volume_mean)

                    from microbiome._vendor.analysis_toolbox.parallel import start_cluster
                    import time

                    with (
                        start_cluster(
                            cores=4,
                            force_local=True,
                            processes=os.cpu_count() // 4,
                            death_timeout=1200,
                            store_logs=True,
                            spillover=True,
                            verbose=True,
                        ) as cluster,
                        Client(cluster) as client,
                    ):

                        display(client)

                        self.im_CM0_dff.volume_mean = (
                            self.im_CM0_dff.volume_mean.compute()
                        )

                        if debug:
                            plt.figure()
                            bg_max_proj = np.nanmax(
                                getattr(self, "im_CM0_dff").volume_mean, axis=0
                            )
                            bg_vmin = np.nanpercentile(bg_max_proj, 30)
                            bg_vmax = np.nanpercentile(bg_max_proj, 99)
                            plt.imshow(
                                np.nanmax(
                                    getattr(spim_dset, spim_dset.im_mode).volume_mean,
                                    axis=0,
                                ),
                                cmap="gray",
                                vmin=bg_vmin,
                                vmax=bg_vmax,
                            )
