# Vendored from analysis_toolbox (dataorg/dataset_collection.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm


def initialize_dataset_collection(setup="ahrens", *args, **kwargs):

    if setup == "ahrens":

        return ahrens_dataset_collection(*args, **kwargs)

    elif setup == "wee":

        return wee_dataset_collection(*args, **kwargs)


class dataset_collection(object):

    def __init__(
        self, data_dirs=[], data_labels=[], collection="collection", debug=False
    ):

        self.data_dirs = data_dirs
        self.data_labels = data_labels
        self.collection = collection

    def init_collection(self, root_dir=""):

        from microbiome._vendor.analysis_toolbox.utils import common_root, path_slash

        if root_dir:
            self.common_path = path_slash(root_dir) + self.collection + "/"
        else:
            self.common_path = (
                common_root(self.data_paths_str) + "/" + self.collection + "/"
            )

        self.procpath = self.common_path + "processed/"
        self.savepath = self.common_path + "analysis/"

        from pathlib import Path

        Path(self.savepath).mkdir(parents=True, exist_ok=True)
        Path(self.procpath).mkdir(parents=True, exist_ok=True)

    def open_datasets(self):
        """Open datasets
        Open datasets and configure the paths for all dataset root
        directories provided.
        """

        from microbiome._vendor.analysis_toolbox.dataorg.dataset import dataset

        if hasattr(self, "data_servers"):
            data_servers = self.data_servers
        else:
            data_servers = [""] * len(self.data_dirs)

        self.dsets = [
            dataset(data_dir, data_server, verbose=False)
            for data_dir, data_server in zip(self.data_dirs, data_servers)
        ]

    def configure_funcim(
        self,
        ephys="raw",
        dataset_type="spim",
        im_raw=False,
        im_pro=False,
        segmented=False,
        align=True,
    ):

        from microbiome._vendor.analysis_toolbox.dataset.functional_imaging import configure_func_im

        if ephys:
            list(map(lambda x: x.init_ephys(), tqdm(self.dsets)))
        if segmented:
            list(map(lambda x: x.init_segments(), tqdm(self.dsets)))

        self.spim_dsets = list(
            map(
                lambda dset: configure_func_im(
                    dset,
                    ephys=ephys,
                    dataset_type=dataset_type,
                    im_raw=im_raw,
                    im_pro=im_pro,
                    segmented=segmented,
                    align=align,
                ),
                tqdm(self.dsets),
            )
        )

    def configure_spim(
        self,
        seg_root=None,
        ephys="raw",
        im_raw=False,
        im_pro=False,
        segmented=False,
        subset=slice(None),
        debug=False,
    ):
        """Initialize and configure datasets for eventual loading of
        ephys and segments

        seg_root [list or string or None]: if string or None, will be converted
                                           to list
        Convenience function
        """
        from microbiome._vendor.analysis_toolbox.dataset.functional_imaging import configure_func_im

        from itertools import starmap

        if not isinstance(seg_root, list):
            seg_root = [seg_root] * len(self.dsets)

        if debug:
            list(
                starmap(
                    lambda x, y: print(x, y),
                    tqdm(
                        zip(self.dsets[subset], seg_root[subset]),
                        total=len(self.dsets[subset]),
                    ),
                )
            )

        if ephys:
            list(map(lambda x: x.init_ephys(), tqdm(self.dsets[subset])))

        if segmented:
            list(
                starmap(
                    lambda x, y: (
                        x.init_segments(seg_root=y)
                        if y is not None
                        else x.init_segments()
                    ),
                    tqdm(
                        zip(self.dsets[subset], seg_root[subset]),
                        total=len(self.dsets[subset]),
                    ),
                )
            )

        self.spim_dsets = list(
            map(
                lambda dset: configure_func_im(
                    dset,
                    dataset_type="spim",
                    ephys=ephys,
                    im_pro=im_pro,
                    im_raw=im_raw,
                    segmented=segmented,
                ),
                tqdm(self.dsets[subset]),
            )
        )

    def configure_ephys(self):
        """Initialize and configure datasets for eventual loading of ephys

        Convenience function
        """

        from microbiome._vendor.analysis_toolbox.dataset.functional_imaging import configure_func_im

        list(map(lambda x: x.init_ephys(), tqdm(self.dsets)))
        self.spim_dsets = list(
            map(lambda dset: configure_func_im(dset, ephys=True), tqdm(self.dsets))
        )

    def init_spim(self, subset=slice(None)):

        list(map(lambda x: x.init(), tqdm(self.spim_dsets[subset])))

        if self.data_labels:

            if isinstance(self.data_labels, list):
                for spim_dset, label in zip(
                    self.spim_dsets[subset], self.data_labels[subset]
                ):
                    spim_dset.label = label

            elif isinstance(self.data_labels, dict):
                for key, value in self.data_labels.items():
                    for spim_dset, label in zip(self.spim_dsets[subset], value[subset]):
                        setattr(spim_dset, key, label)

    def init_seg(self):
        """
        Only applies to single plane recordings
        """

        # TODO: hard coded im_CM0_seg, should be able to find all seg objects
        list(
            map(lambda x: x.im_CM0_seg.load_brain_vol(), self.spim_dsets)
        )  # done when segments are loaded
        list(map(lambda x: x.im_CM0_seg.find_segment_centroids(), self.spim_dsets))

        for spim_dset in tqdm(self.spim_dsets):
            spim_dset.im_CM0_seg.image_mean = spim_dset.im_CM0_seg.volume_mean[
                np.newaxis, :, :
            ]

    def init_collab(self, to_replace, replace_with):

        list(
            map(
                lambda x: link_collab_dataset(x, to_replace, replace_with),
                tqdm(self.dsets),
            )
        )

    def create_metadata(self):

        list(map(lambda x: x.create_metadata(), tqdm(self.dsets)))

    def return_dataset_pair(self, dataset_pair):

        from microbiome._vendor.analysis_toolbox.utils import return_object_containing_attr_of_value

        excluded_expt_id, included_expt_id = dataset_pair

        excluded_spim_dset = return_object_containing_attr_of_value(
            self.spim_dsets, "expt_id", excluded_expt_id
        )
        included_spim_dset = return_object_containing_attr_of_value(
            self.spim_dsets, "expt_id", included_expt_id
        )

        return excluded_spim_dset, included_spim_dset

    def find_interval_between_spim_dsets(self, first_spim_dset, second_spim_dset):

        from datetime import datetime, timedelta

        first_spim_dset.expt_start_datetime = datetime.strptime(
            first_spim_dset.ep.metadata_pd.loc["timestamp", "value"],
            "%I:%M:%S.%f %p %m/%d/%Y",
        )
        first_spim_dset.expt_end_datetime = (
            first_spim_dset.expt_start_datetime
            + timedelta(seconds=first_spim_dset.duration)
        )
        second_spim_dset.expt_start_datetime = datetime.strptime(
            second_spim_dset.ep.metadata_pd.loc["timestamp", "value"],
            "%I:%M:%S.%f %p %m/%d/%Y",
        )
        assert first_spim_dset.expt_end_datetime < second_spim_dset.expt_start_datetime

        datetime_interval = (
            second_spim_dset.expt_start_datetime - first_spim_dset.expt_end_datetime
        )
        datetime_diff = (
            second_spim_dset.expt_start_datetime - first_spim_dset.expt_start_datetime
        )

        return datetime_diff

    def combine_datasets(self, dataset_pair=()):

        import types

        combined_dset = types.SimpleNamespace()
        combined_spim_dset = types.SimpleNamespace()

        excluded_spim_dset, included_spim_dset = self.return_dataset_pair(dataset_pair)
        datetime_diff = self.find_interval_between_spim_dsets(
            included_spim_dset, excluded_spim_dset
        )

        combined_spim_dset.ep = types.SimpleNamespace()
        setattr(
            combined_spim_dset.ep,
            "ep",
            np.concatenate(
                [included_spim_dset.ep.ep, excluded_spim_dset.ep.ep], axis=1
            ),
        )
        setattr(
            combined_spim_dset.ep,
            "t_s",
            np.concatenate(
                [
                    included_spim_dset.ep.t_s,
                    excluded_spim_dset.ep.t_s + datetime_diff.total_seconds(),
                ]
            ),
        )
        setattr(
            combined_spim_dset,
            "t_axis",
            np.concatenate(
                [
                    included_spim_dset.t_axis,
                    excluded_spim_dset.t_axis + datetime_diff.total_seconds(),
                ]
            ),
        )

        setattr(
            combined_spim_dset,
            "spim_dset_origin",
            (included_spim_dset, excluded_spim_dset),
        )
        setattr(
            combined_spim_dset,
            "expt_id",
            (included_spim_dset.expt_id, excluded_spim_dset.expt_id),
        )
        setattr(
            combined_spim_dset,
            "savepaths",
            (included_spim_dset.savepath, excluded_spim_dset.savepath),
        )
        setattr(
            combined_spim_dset,
            "procpaths",
            (included_spim_dset.procpath, excluded_spim_dset.procpath),
        )

        return combined_dset, combined_spim_dset

    def fix_time_column(self, dataset_pairs={}, output_datasets=[]):

        for dataset_pair, output_dataset in zip(dataset_pairs.items(), output_datasets):
            excluded_expt_id, included_expt_id = dataset_pair
            excluded_spim_dset, included_spim_dset = self.return_dataset_pair(
                dataset_pair
            )
            datetime_diff = self.find_interval_between_spim_dsets(
                included_spim_dset, excluded_spim_dset
            )

            output_dataset.trials_df.loc[
                output_dataset.trials_df["expt_id"] == excluded_spim_dset.expt_id,
                "time",
            ] = (
                output_dataset.trials_df.loc[
                    output_dataset.trials_df["expt_id"] == excluded_spim_dset.expt_id,
                    "time",
                ]
                + datetime_diff.total_seconds()
            )

            output_dataset.trials_df.loc[
                output_dataset.trials_df["expt_id"] == excluded_spim_dset.expt_id,
                "trial_start_time",
            ] = (
                output_dataset.trials_df.loc[
                    output_dataset.trials_df["expt_id"] == excluded_spim_dset.expt_id,
                    "trial_start_time",
                ]
                + datetime_diff.total_seconds()
            )

    def port_attr_over(self, input_dataset_ids, output_datasets, attr):

        if isinstance(input_dataset_ids, dict):
            input_dataset_ids = list(input_dataset_ids.items())

        assert isinstance(input_dataset_ids, list)

        for input_dataset_id, output_dataset in zip(input_dataset_ids, output_datasets):

            from microbiome._vendor.analysis_toolbox.utils import return_object_containing_attr_of_value

            if isinstance(output_dataset, str):

                output_dataset = return_object_containing_attr_of_value(
                    self.spim_dsets, "expt_id", output_dataset
                )

            if isinstance(input_dataset_id, tuple):

                _, input_dataset = self.return_dataset_pair(input_dataset_id)
                setattr(output_dataset, attr, getattr(input_dataset, attr))

            elif isinstance(input_dataset_id, str):

                input_dataset = return_object_containing_attr_of_value(
                    self.spim_dsets, "expt_id", input_dataset_id
                )

            setattr(output_dataset, attr, getattr(input_dataset, attr))

    def combine_dataset_pairs(self, dataset_pairs={}):

        combined_dsets = []
        combined_spim_dsets = []

        for dataset_pair in dataset_pairs.items():

            combined_dset, combined_spim_dset = self.combine_datasets(dataset_pair)

            combined_dsets.append(combined_dset)
            combined_spim_dsets.append(combined_spim_dset)

        return combined_dsets, combined_spim_dsets

    def combine_excluded_datasets(self, inplace=False, debug=False):

        self.combined_dsets, self.combined_spim_dsets = self.combine_dataset_pairs(
            dataset_pairs=self.excluded_datasets
        )

        dsets = self.dsets + self.combined_dsets
        spim_dsets = self.spim_dsets + self.combined_spim_dsets

        src_dset_bool = [1] * len(self.dsets) + [0] * len(self.combined_dsets)

        if inplace:

            if debug:
                print(f"Permanently modifying dsets and spim_dsets")

            self.dsets = dsets
            self.spim_dsets = spim_dsets
            self.src_dset_bool = src_dset_bool

        else:
            return dsets, spim_dsets

    def combine_dataframes(self, df_name, attr_list=['expt_id'], df_out="", dataset_pairs={}, set_attr=False):
        """
        set_attr [bool or list of combined_spim_dsets]:
            if set_attr is True: setattr to included_spim_dset
            if set_attr is False: returns the new dataframe
            if set_attr is a list: setattr to objects in the list
        """

        df_collection = []

        if 'expt_id' not in attr_list:
            attr_list += ['expt_id']

        if dataset_pairs:

            for excluded_expt_id, included_expt_id in dataset_pairs.items():

                excluded_spim_dset, included_spim_dset = self.return_dataset_pair(
                    (excluded_expt_id, included_expt_id)
                )

                df_collection = []

                for spim_dset in [included_spim_dset, excluded_spim_dset]:

                    df = getattr(spim_dset, df_name).copy()
                    for attr in attr_list: df[attr] = getattr(spim_dset, attr)
                    df.reset_index(inplace=True)
                    df_collection.append(df)

        else:

            for dset, spim_dset in tqdm(
                zip(self.dsets, self.spim_dsets), total=len(self.spim_dsets)
            ):

                df = getattr(spim_dset, df_name).copy()
                for attr in attr_list: df[attr] = getattr(spim_dset, attr)
                df.reset_index(inplace=True)
                df_collection.append(df)

        if len(df_collection) > 0:

            collection_df = pd.concat(df_collection, axis=0, ignore_index=True)

            if set_attr:

                if not df_out:
                    df_out = df_name

                if dataset_pairs:

                    for i, (excluded_expt_id, included_expt_id) in enumerate(
                        dataset_pairs.items()
                    ):

                        if isinstance(set_attr, list):

                            setattr(set_attr[i], df_out, collection_df)

                        elif isinstance(set_attr, bool):

                            excluded_spim_dset, included_spim_dset = (
                                self.return_dataset_pair(
                                    (excluded_expt_id, included_expt_id)
                                )
                            )

                            ## save a copy of trials_df
                            setattr(
                                excluded_spim_dset,
                                f"{df_name}_original",
                                getattr(excluded_spim_dset, df_name),
                            )
                            setattr(
                                included_spim_dset,
                                f"{df_name}_original",
                                getattr(included_spim_dset, df_name),
                            )

                            setattr(included_spim_dset, df_out, collection_df)

                else:

                    setattr(self, df_out, collection_df)

            else:

                return collection_df

    def concatenate_attribute_for_same_fish(
        self, attr_name, dataset_pairs={}, output_objects=[]
    ):

        from microbiome._vendor.analysis_toolbox.utils import rgetattr, rsetattr

        for i, (excluded_expt_id, included_expt_id) in enumerate(dataset_pairs.items()):

            excluded_spim_dset, included_spim_dset = self.return_dataset_pair(
                (excluded_expt_id, included_expt_id)
            )
            output_object = output_objects[i]

            included_spim_dset_attr = rgetattr(included_spim_dset, attr_name)
            excluded_spim_dset_attr = rgetattr(excluded_spim_dset, attr_name)

            if isinstance(included_spim_dset_attr, np.ndarray):

                print(f"Rececived attribute of type {type(included_spim_dset_attr)}")

                rsetattr(
                    output_object,
                    attr_name,
                    np.concatenate([included_spim_dset_attr, excluded_spim_dset_attr]),
                )

            elif isinstance(included_spim_dset_attr, dict):

                print(f"Rececived attribute of type {type(included_spim_dset_attr)}")

                output_object_attr = {}

                for key in included_spim_dset_attr.keys():

                    output_object_attr[key] = np.concatenate(
                        [included_spim_dset_attr[key], excluded_spim_dset_attr[key]]
                    )

                rsetattr(output_object, attr_name, output_object_attr)

    def combine_attribute_for_same_fish(
        self,
        attr_name,
        dataset_pairs={},
        combine_mode="combine",
        output_objects=[],
        force=False,
    ):

        for i, (excluded_expt_id, included_expt_id) in enumerate(dataset_pairs.items()):

            excluded_spim_dset, included_spim_dset = self.return_dataset_pair(
                (excluded_expt_id, included_expt_id)
            )
            output_object = output_objects[i]

            self.combine_attribute(
                attr_name,
                combine_mode=combine_mode,
                input_objects=[excluded_spim_dset, included_spim_dset],
                output_object=output_object,
                force=force,
            )

    def combine_attribute(
        self,
        attr_name,
        attr_out_name="",
        combine_mode="combine",
        input_objects=None,
        output_object=None,
        force=False,
    ):

        from microbiome._vendor.analysis_toolbox.utils import combine_attribute

        if input_objects is None:
            input_objects = self.spim_dsets

        if isinstance(input_objects[0], str):
            from microbiome._vendor.analysis_toolbox.utils import return_object_containing_attr_of_value

            input_objects = [
                return_object_containing_attr_of_value(
                    self.spim_dsets, "expt_id", input_object
                )
                for input_object in input_objects
            ]

        return combine_attribute(
            input_objects=input_objects,
            attr_name=attr_name,
            combine_mode=combine_mode,
            output_object=output_object,
            attr_out_name=attr_out_name,
            force=force,
        )

    def search_for_same_fish(self):

        self.fish_num = [
            spim_dset.fish_num
            for dset, spim_dset in tqdm(
                zip(self.dsets, self.spim_dsets), total=len(self.spim_dsets)
            )
        ]
        self.expt_date = [
            spim_dset.expt_date
            for dset, spim_dset in tqdm(
                zip(self.dsets, self.spim_dsets), total=len(self.spim_dsets)
            )
        ]
        self.expt_num = [
            spim_dset.expt_num
            for dset, spim_dset in tqdm(
                zip(self.dsets, self.spim_dsets), total=len(self.spim_dsets)
            )
        ]

        from microbiome._vendor.analysis_toolbox.utils import zip_and_join, find_duplicates

        self.fish_expt_nums = zip_and_join(self.expt_date, self.fish_num, sep=None)
        self.fish_with_multiple_datasets = find_duplicates(self.fish_expt_nums)

        for dset, spim_dset, fish_expt_nums in tqdm(
            zip(self.dsets, self.spim_dsets, self.fish_expt_nums),
            total=len(self.spim_dsets),
        ):

            spim_dset.fish_expt_nums = fish_expt_nums

        from collections import OrderedDict

        self.excluded_datasets = OrderedDict()

        for fish in self.fish_with_multiple_datasets:

            dataset_list = []

            for dset, spim_dset in tqdm(
                zip(self.dsets, self.spim_dsets), total=len(self.spim_dsets)
            ):

                if spim_dset.fish_expt_nums == fish:

                    dataset_list.append(spim_dset)

            ## Dictionary of excluded dataset:included dataset pairs
            for spim_dset in dataset_list[1:]:
                self.excluded_datasets[spim_dset.expt_id] = dataset_list[0].expt_id

        self.excluded_dataset_list = list(self.excluded_datasets.keys()) + list(
            self.excluded_datasets.values()
        )

        print("{: >20} {: >20}".format(*("excluded", "included")))
        for row in self.excluded_datasets.items():
            print("{: >20} {: >20}".format(*row))

    def compute_combined_t_axis(self, fs_attrname, taxis_attrname):
        from microbiome._vendor.analysis_toolbox.utils import rgetattr, crange

        max_im_rate = np.ceil(
            np.nanmax(
                [rgetattr(spim_dset, fs_attrname) for spim_dset in self.spim_dsets]
            )
        )
        min_trace_dur = np.floor(
            np.nanmin(
                [
                    rgetattr(spim_dset, taxis_attrname)[0]
                    for spim_dset in self.spim_dsets
                ]
            )
        )
        max_trace_dur = np.ceil(
            np.nanmax(
                [
                    rgetattr(spim_dset, taxis_attrname)[-1]
                    for spim_dset in self.spim_dsets
                ]
            )
        )
        new_t_axis = crange(min_trace_dur, max_trace_dur, 1 / max_im_rate)

        return new_t_axis

    def interpolate_trace(
        self, t_axis, trace, new_t_axis=None, fs_attrname="", taxis_attrname=""
    ):
        """
        Either new_t_axis or (fs_attrname, taxis_attrname) need to be provided
        """
        from scipy.interpolate import interp1d

        if new_t_axis is None:
            assert fs_attrname
            assert taxis_attrname
            new_t_axis = self.compute_combined_t_axis(fs_attrname, taxis_attrname)

        trace_func = interp1d(t_axis, trace, fill_value="extrapolate")
        new_trace = trace_func(new_t_axis)

        return new_trace

    def create_namespace(self, attr):

        from microbiome._vendor.analysis_toolbox.utils import create_necessary_namespaces

        create_necessary_namespaces(self, attr)

    def return_dataset_with_attributes(self, **kwargs):

        from microbiome._vendor.analysis_toolbox.utils import return_object_with_attributes

        return return_object_with_attributes(self.spim_dsets, **kwargs)
    
    def return_attr_from_dataset_with_attributes(self, attr_name, **kwargs):

        return getattr(self.return_dataset_with_attributes(**kwargs), attr_name)

    def dask_client_compute(self, list_of_futures, list_of_datasets=[], attr_name=""):

        from microbiome._vendor.analysis_toolbox.parallel import dask_client_compute
        from microbiome._vendor.analysis_toolbox.utils import rsetattr

        list_of_results = dask_client_compute(list_of_futures)

        if not len(list_of_datasets) == len(list_of_results):

            list_of_datasets = self.spim_dsets

        for dataset, result in zip(list_of_datasets, list_of_results):

            rsetattr(dataset, attr_name, result)


class wee_dataset_collection(dataset_collection):
    """A collection of analysis_toolbox.dataset.expt.expt objects

    Class with methods that are mostly wrappers for running for loops
    over all the datasets.
    """

    def __init__(
        self, data_dirs=[], data_labels=[], collection="collection", analysis_dir="", debug=False
    ):

        super().__init__(
            data_dirs=data_dirs,
            data_labels=data_labels,
            collection=collection,
            debug=debug,
        )

        self.data_paths_str = self.data_dirs

        self.init_collection(root_dir=analysis_dir)

    def prepare_dir(self):
        """Augment new datasets by creating standard directories and
        metadata file.

        Won't overwrite exisiting directories and metadata file.
        """

        list(map(lambda x: x.prepare_dir(), tqdm(self.dsets)))


class ahrens_dataset_collection(dataset_collection):
    """A collection of analysis_toolbox.dataset.expt.expt objects

    Class with methods that are mostly wrappers for running for loops
    over all the datasets.
    """

    def __init__(
        self,
        datasets=[],
        data_dirs=[],
        data_servers=[],
        collection="collection",
        debug=False,
    ):

        super().__init__(
            data_dirs=data_dirs,
            data_labels=data_labels,
            collection=collection,
            debug=debug,
        )

        if datasets:

            assert all(isinstance(x, dict) for x in datasets)

            self.data_dirs = [dataset["dir"] for dataset in datasets]
            self.data_servers = [dataset["server"] for dataset in datasets]
            self.data_labels = [dataset["label"] for dataset in datasets]

        else:

            if isinstance(data_servers, str):
                data_servers = [data_servers] * len(data_dirs)

            assert all(
                len(lst) == len(data_dirs) for lst in [data_dirs, data_servers]
            ), f"{len(data_dirs)}, {len(data_servers)}"

            self.data_dirs = data_dirs
            self.data_servers = data_servers
            self.data_labels = data_labels

        self.collection = collection

        self.find_data_paths()
        self.init_collection(root_dir="/nrs/ahrens/jing/statemod/statedep/analysis/")

    def find_data_paths(self):

        from microbiome._vendor.analysis_toolbox.janelia_helper import janelia_path

        self.data_paths = [
            janelia_path(data_dir, server=data_server)
            for data_dir, data_server in zip(self.data_dirs, self.data_servers)
        ]
        self.data_paths_str = [data_path.as_posix() for data_path in self.data_paths]

    def prepare_dir(self):
        """Augment new datasets by creating standard directories and
        metadata file.

        Won't overwrite exisiting directories and metadata file.
        """

        list(map(lambda x: x.prepare_dir(), tqdm(self.dsets)))
        list(map(lambda x: x.create_metadata(), tqdm(self.dsets)))
        list(map(lambda x: x.add_stack_freq_to_metadata(), tqdm(self.dsets)))


def link_collab_dataset(dset, to_replace, replace_with):

    from microbiome._vendor.analysis_toolbox.utils import symlink

    for im_dir_str in dset.im_dir_str:

        symlink(
            im_dir_str.replace(to_replace, replace_with) + "ch0.xml",
            im_dir_str + "ch0.xml",
            overwrite=True,
        )

    symlink(
        dset.ep_dir_str.replace(to_replace, replace_with)
        + "_".join(dset.expt_name.split("_")[:-2])
        + ".10chFlt",
        dset.ep_dir_str + "_".join(dset.expt_name.split("_")[:-2]) + ".10chFlt",
        overwrite=True,
    )

    for seg_dir_str in dset.seg_dir_str:

        symlink(
            seg_dir_str.replace(to_replace, replace_with) + "cell_dff.npz",
            seg_dir_str + "cell_dff.npz",
            overwrite=True,
        )
        symlink(
            seg_dir_str.replace(to_replace, replace_with) + "Y_ave.npy",
            seg_dir_str + "Y_ave.npy",
            overwrite=True,
        )
