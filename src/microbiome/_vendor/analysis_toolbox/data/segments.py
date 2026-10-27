# Vendored from analysis_toolbox (data/segments.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


from microbiome._vendor.analysis_toolbox.data.imaging import voxel
import os
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import pandas as pd
import zarr
from tqdm.auto import tqdm
import h5py


class voluseg(voxel):
    def __init__(
        self,
        segmented_root="",
        im_dir="",
        external_segmented_root="",
        # load options
        mode="simple",
        # extra options
        metadata="",
        debug=False,
    ):

        ## Sets imaging related variables #################
        # self.path, self.debug, self.metadata_dir,
        # self.savepath, self.procpath,
        # self.expt_id, self.expt_date, self.expt_name,
        # self.metadata, self.metadata_pd
        ###################################################
        # if segmented is not a subdir of its original directory
        # (e.g. when it is in Ziqiang's folder), we still want to save
        # analysis/results etc in the original directory
        if im_dir:

            super().__init__(im_dir=im_dir, metadata=metadata, debug=debug)

        # if segmented is in a subdir of its original directory,
        # we can use this directory to find the right subdirs to save
        # analysis/results etc
        else:

            super().__init__(im_dir=segmented_root, metadata=metadata, debug=debug)

        self.segmented_root = segmented_root
        self.external_segmented_root = external_segmented_root
        self.mode = mode
        self.timeseries_file = f"{self.segmented_root}mean_timeseries.hdf5"
        self.cell_file = f"{self.segmented_root}cells_clean.hdf5"
        self.cell_dff_file = f"{self.segmented_root}cells_clean_dff.zarr"
        self.volume_file = f"{self.segmented_root}volume.hdf5"

        # Load segmented files
        if self.mode == "full":
            self.load_segmented_files()
            self.calculate_dff()

        elif self.mode == "simple":
            self.load_brain_vol()
            if os.path.exists(self.cell_dff_file):
                self.load_segments_dff()
            self.load_segments()
            if not hasattr(self, "cell_dff"):
                self.calculate_dff()

        elif self.mode == "loc":
            self.load_brain_vol()
            self.load_segment_loc()

        elif self.mode == "nothing":
            pass

        self.compute_simple_variables()

    def load_segmented_files(self):

        self.load_cell_file()
        self.load_volume_file()
        self.load_timeseries_file()

    def load_segments_dff(self, dFF="cell_dff"):

        print("Loading dff of segments...")

        import zarr

        setattr(self, f"{dFF}_zarr", zarr.open(self.cell_dff_file))
        print("Loaded dff for segments! Printing attributes...")
        display(getattr(self, f"{dFF}_zarr").attrs.asdict())
        ## 12.5s for array of shape (345466, 10805)
        setattr(self, dFF, getattr(self, f"{dFF}_zarr")[:])

    def load_segments(self):
        print("Loading segments...")

        import h5py

        with h5py.File(self.cell_file, "r") as hf:
            # estimated background fluorescence [np.float64]
            self.background = hf["background"][()]

            if not hasattr(self, "cell_dff"):
                # detrend (and optionally filtered) timeseries [np.ndarray with shape (ncells, ntimepoints)]
                self.cell_timeseries = hf["cell_timeseries"][()]
                # computed cell baselines [np.ndarray with shape (ncells, ntimepoints)]
                self.cell_baseline = hf["cell_baseline"][()]

            # cell spatial footprints (spatial NMF components) [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_weights = hf["cell_weights"][()]
            # cell x coordinates [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_x = hf["cell_x"][()]
            # cell y coordinates [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_y = hf["cell_y"][()]
            # cell y coordinates [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_z = hf["cell_z"][()]
            self.n = hf["n"][()]  # number of cells [np.int64]
            self.t = hf["t"][()]  # number of timepoints [np.int64]
            self.x = hf["x"][()]  # x volume dimension [np.int64]
            self.y = hf["y"][()]  # y volume dimension [np.int64]
            self.z = hf["z"][()]  # z volume dimension [np.int64]

    def load_segment_loc(self):
        print("Loading segments...")

        import h5py

        with h5py.File(self.cell_file, "r") as hf:

            self.background = hf["background"][
                ()
            ]  # estimated background fluorescence [np.float64]
            self.cell_weights = hf["cell_weights"][
                ()
            ]  # cell spatial footprints (spatial NMF components) [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_x = hf["cell_x"][
                ()
            ]  # cell x coordinates [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_y = hf["cell_y"][
                ()
            ]  # cell y coordinates [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_z = hf["cell_z"][
                ()
            ]  # cell y coordinates [np.ndarray with shape (ncells, ncoordinates)]
            self.n = hf["n"][()]  # number of cells [np.int64]
            self.t = hf["t"][()]  # number of timepoints [np.int64]
            self.x = hf["x"][()]  # x volume dimension [np.int64]
            self.y = hf["y"][()]  # y volume dimension [np.int64]
            self.z = hf["z"][()]  # z volume dimension [np.int64]

    def load_brain_vol(self):
        print("Loading brain volume...")

        import h5py

        with h5py.File(self.volume_file, "r") as hf:

            self.volume_mean = hf["volume_mean"][
                ()
            ]  # volume mean [np.ndarray with shape (x, y, z)]
            self.volume_mask = hf["volume_mask"][
                ()
            ]  # volume mask [np.ndarray with shape (x, y, z)]

    def calculate_dff(
        self,
        dFF="cell_dff",
        F="cell_timeseries",
        F_baseline="cell_baseline",
        background="background",
        offset=0,
        store=True,
    ):
        """
        offset used to be 10 (taken from Davis' code)
        now set to 0 to amplify dff
        """

        if isinstance(F, str):
            F = getattr(self, F)
        if isinstance(F_baseline, str):
            F_baseline = getattr(self, F_baseline)
        if isinstance(background, str):
            background = getattr(self, background)

        print("Calculating dff...")

        from microbiome._vendor.analysis_toolbox.data.voluseg import calculate_dff

        setattr(self, dFF, calculate_dff(F, F_baseline, background, offset=offset))

        if store:
            self.store_dff(dFF=dFF, offset=offset)

    def store_dff(self, dFF="cell_dff", offset=0):

        from microbiome._vendor.analysis_toolbox.utils import get_chunksize_from_chunks

        ## 27.5s for array of shape (345466, 10805)
        chunksize = get_chunksize_from_chunks(
            getattr(self, dFF), mode="full", chunks=("auto", -1)
        )

        import zarr

        store = zarr.DirectoryStore(self.cell_dff_file)
        ## 47.1 s for array of shape (345466, 10805)
        setattr(
            self,
            f"{dFF}_dff",
            zarr.array(getattr(self, dFF), chunks=chunksize, store=store),
        )

        getattr(self, f"{dFF}_dff").attrs["offset"] = offset

    def load_cell_file(self):

        import h5py

        with h5py.File(self.cell_file, "r") as hf:

            self.background = hf["background"][
                ()
            ]  # estimated background fluorescence [np.float64]
            self.cell_block_id = hf["cell_block_id"][
                ()
            ]  # [np.ndarray with shape (ncells,)]
            self.cell_baseline = hf["cell_baseline"][
                ()
            ]  # computed cell baselines [np.ndarray with shape (ncells, ntimepoints)]
            self.cell_timeseries = hf["cell_timeseries"][
                ()
            ]  # detrend (and optionally filtered) timeseries [np.ndarray with shape (ncells, ntimepoints)]
            self.cell_timeseries_raw = hf["cell_timeseries_raw"][
                ()
            ]  # raw cell_timeseries (direct output of segmentation) [np.ndarray with shape (ncells, ntimepoints)]
            self.cell_weights = hf["cell_weights"][
                ()
            ]  # cell spatial footprints (spatial NMF components) [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_x = hf["cell_x"][
                ()
            ]  # cell x coordinates [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_y = hf["cell_y"][
                ()
            ]  # cell y coordinates [np.ndarray with shape (ncells, ncoordinates)]
            self.cell_z = hf["cell_z"][
                ()
            ]  # cell y coordinates [np.ndarray with shape (ncells, ncoordinates)]
            self.n = hf["n"][()]  # number of cells [np.int64]
            self.t = hf["t"][()]  # number of timepoints [np.int64]
            self.volume_id = hf["volume_id"][
                ()
            ]  # cell ids represented on a volume [np.ndarray with shape (x, y, z)]
            self.volume_weight = hf["volume_weight"][
                ()
            ]  # cell spatial footprints represented on a volume [np.ndarray with shape (x, y, z)]
            self.x = hf["x"][()]  # x volume dimension [np.int64]
            self.y = hf["y"][()]  # y volume dimension [np.int64]
            self.z = hf["z"][()]  # z volume dimension [np.int64]

    def load_volume_file(self):

        import h5py

        with h5py.File(self.volume_file, "r") as hf:

            self.background = hf["background"][
                ()
            ]  # estimated background fluorescence [np.float64]
            self.block_valids = hf["block_valids"][
                ()
            ]  # indices of blocks used for segmentation [np.ndarray with shape (nblocks,)]
            self.block_xyz0 = hf["block_xyz0"][
                ()
            ]  # min block xyz coordinates [np.ndarray with shape (nblocks, ndim)]
            self.block_xyz1 = hf["block_xyz1"][
                ()
            ]  # max block xyz coordinates [np.ndarray with shape (nblocks, ndim)]
            # total number of blocks  [np.int64]
            self.n_blocks = hf["n_blocks"][()]
            self.n_voxels_cell = hf["n_voxels_cell"][
                ()
            ]  # approximate number of voxels in each cell [np.int64]
            self.thr_intensity = hf["thr_intensity"][
                ()
            ]  # brain-mask intensity threshold [np.float64]
            self.thr_probability = hf["thr_probability"][
                ()
            ]  # brain-mask probability threshold [np.float64]
            self.volume_mask = hf["volume_mask"][
                ()
            ]  # volume mask [np.ndarray with shape (x, y, z)]
            self.volume_mean = hf["volume_mean"][
                ()
            ]  # volume mean [np.ndarray with shape (x, y, z)]
            self.volume_peak = hf["volume_peak"][
                ()
            ]  # volume peak intensity [np.ndarray with shape (x, y, z)]

    def load_timeseries_file(self):

        import h5py

        with h5py.File(self.timeseries_file, "r") as hf:

            self.mean_baseline = hf["mean_baseline"][
                ()
            ]  # mean baseline [np.ndarray of shape (1, ntimepoints)]
            self.mean_timeseries = hf["mean_timeseries"][
                ()
            ]  # mean timeseries [np.ndarray of shape (1, ntimepoints)]
            self.mean_timeseries_raw = hf["mean_timeseries_raw"][
                ()
            ]  # mean timeseries raw [np.ndarray of shape (1, ntimepoints)]
            self.timepoints = hf["timepoints"][
                ()
            ]  # indices of timepionts used for cell segmentation

    def load_old_segmented_files(self, debug=False):

        self.load_old_mask_file()
        self.load_old_cell_file()
        self.load_old_cleaned_file()
        self.load_old_component_file()

        print("Segmented imaging data loaded!")

    def load_old_mask_file(self, debug=False):

        try:

            f = h5py.File(self.mask_file, "r")

            self.background = f["background"][()]
            self.blok_lidx = f["blok_lidx"][()]
            self.blok_nmbr = f["blok_nmbr"][()]
            self.brain_mask = f["brain_mask"][()]
            self.image_mean = f["image_mean"][()]
            self.image_peak = f["image_peak"][()]
            self.image_peak_fine = f["image_peak_fine"][()]
            self.thr_mask = f["thr_mask"][()]
            self.thr_prob = f["thr_prob"][()]

            f.close()

        except:
            print("Problems loading %s" % self.mask_file)

    def load_old_cell_file(self, debug=False):

        try:
            f = h5py.File(self.cell_file, "r")

            self.Cmpn_position = f["Cmpn_position"][()]
            self.Cmpn_spcesers = f["Cmpn_spcesers"][()]
            self.Cmpn_timesers = f["Cmpn_timesers"][()]
            self.dims = f["dims"][()]
            self.freq = int(f["freq"][()])
            self.resn = f["resn"][()]

            f.close()

        except:
            print("Problems loading %s" % self.cell_file)

    def load_old_cleaned_file(self, debug=False):

        try:
            f = h5py.File(self.cleaned_file, "r")

            self.Cell_X = f["Cell_X"][()]
            self.Cell_Y = f["Cell_Y"][()]
            self.Cell_Z = f["Cell_Z"][()]
            self.Cell_baseline1 = f["Cell_baseline1"][()]
            self.Cell_spcesers = f["Cell_spcesers"][()]
            self.Cell_timesers0 = f["Cell_timesers0"][()]
            self.Cell_timesers1 = f["Cell_timesers1"][()]
            self.Labels = f["Labels"][()]
            self.Volume = f["Volume"][()]
            self.background = f["background"][()]
            self.freq = int(f["freq"][()])
            self.n = int(f["n"][()])
            self.x = int(f["x"][()])
            self.y = int(f["y"][()])
            self.z = int(f["z"][()])

            f.close()

        except:
            print("Problems loading %s" % self.cleaned_file)

    def load_old_component_file(self, debug=False):

        try:
            f = h5py.File(self.component_file, "r")

            self.H0 = f["H0"][()]
            self.W0 = f["W0"][()].transpose()

            try:
                self.H1 = f["H1"][()]
                self.W1 = f["W1"][()].transpose()
            except:
                pass

            f.close()

        except:
            print("Problems loading %s" % self.component_file)

    def put_segments_in_df(self):

        if hasattr(self, "cell_weights"):
            seg_dict = {
                "cell_weights": list(self.cell_weights),
                "cell_x": list(self.cell_x),
                "cell_y": list(self.cell_y),
                "cell_z": list(self.cell_z),
            }

            if hasattr(self, "cell_dff"):

                seg_dict["dff"] = list(self.cell_dff)

            setattr(self, "seg_df", pd.DataFrame(seg_dict))

    def find_z(self, seg):

        big_weights_thresh = np.nanpercentile(seg["cell_weights"], 75)
        big_weights_bool = seg["cell_weights"] > big_weights_thresh

        from scipy import stats

        mode_z, count_z = stats.mode(
            seg["cell_z"][big_weights_bool], axis=None, nan_policy="omit"
        )

        return mode_z

    def compute_simple_variables(self):

        # compute commonly used variables
        if hasattr(self, "z") and hasattr(self, "y") and hasattr(self, "x"):
            self.dset_dims = np.array([self.z, self.y, self.x])

        if hasattr(self, "raw_vox_spacing"):
            self.compute_vox_spacing()

    def find_segment_centroids(self):
        """Find the location of cells"""

        new_cols = ["Centroid x", "Centroid y", "Centroid z"]

        self.seg_df[new_cols] = self.seg_df.progress_apply(
            find_segment_centroids, axis=1
        )

    def find_cell(
        self,
        cell_num,
        cell_volume=None,
        mask=1,
        pixbypix=False,
        z_offset=0,
        y_offset=0,
        x_offset=0,
    ):

        from microbiome._vendor.analysis_toolbox.data.voluseg import find_cell

        return find_cell(
            cell_num,
            self.cell_z,
            self.cell_y,
            self.cell_x,
            self.cell_weights,
            (self.z, self.y, self.x),
            cell_volume=cell_volume,
            mask=mask,
            pixbypix=pixbypix,
            z_offset=z_offset,
            y_offset=y_offset,
            x_offset=x_offset,
        )

    def normalize_cell_weights(self):

        row_wghts = np.nanmax(self.cell_weights, axis=1, keepdims=True)
        norm_wghts = self.cell_weights / row_wghts

        self.norm_cell_weights = norm_wghts

    def find_cells(
        self,
        cell_num,
        cell_volume=None,
        mask=1,
        pixbypix=False,
        z_offset=0,
        y_offset=0,
        x_offset=0,
        weight_multiplier=1,
        debug=False,
    ):

        if not hasattr(self, "norm_cell_weights"):
            self.normalize_cell_weights()

        from microbiome._vendor.analysis_toolbox.data.voluseg import find_cells

        cell_volume = find_cells(
            cell_num,
            self.cell_z,
            self.cell_y,
            self.cell_x,
            self.norm_cell_weights * weight_multiplier,
            (self.z, self.y, self.x),
            cell_volume=cell_volume,
            mask=mask,
            pixbypix=pixbypix,
            z_offset=z_offset,
            y_offset=y_offset,
            x_offset=x_offset,
        )

        return cell_volume

    def initialize_regd(self, force=None, debug=False):

        if (
            os.path.exists(self.procpath + "registered_cell_locs.npz")
            or os.path.exists(self.segmented_root + "registered_cell_locs.npz")
        ) and force != "weiz":

            print("Found processed files for Javier's registration...")

            self.regd = voluseg_regd(self)

        elif os.path.exists(self.external_segmented_root) and force != "howj":

            print("Found processed files in Ziqiang's directory")

            from statemod.givingup_helper import statemod_cellProcessing_regd

            self.regd = statemod_cellProcessing_regd(self.external_segmented_root)
            self.regd.load_all_cell_centers(debug=debug)
            self.regd.index_brain_cell_centers()
            self.regd.load_atlas_labels()

    def compute_cell_anat(self, num_of_processes=None, persist=False):

        if not hasattr(self, "regd"):

            self.initialize_regd()

        from statemod.givingup_helper import statemod_cellProcessing_regd

        if isinstance(self.regd, voluseg_regd):

            self.regd.load_registered_seg_loc()
            self.regd.load_reference_brain()
            self.regd.setup_seg_df()
            self.regd.find_segment_centroids()
            self.all_cells_zyx = self.regd.seg_df[
                ["Centroid z", "Centroid y", "Centroid x"]
            ].to_numpy()

        elif isinstance(self.regd, statemod_cellProcessing_regd):

            if hasattr(self.regd, "registered_cell_centers_raw_space"):
                self.all_cells_zyx = self.regd.registered_cell_centers_raw_space
            else:
                self.all_cells_zyx = self.regd.registered_cell_centers

        print(f"Computing for {self.expt_id}...")

        from microbiome._vendor.analysis_toolbox.parallel import parallelize, num_cpu
        from microbiome._vendor.analysis_toolbox.data.segments import (
            compute_cell_location_for_multiple_cells,
        )

        if num_of_processes is None:
            num_of_processes = num_cpu()

        self.all_cells_locs = parallelize(
            self.all_cells_zyx.astype("int"),
            compute_cell_location_for_multiple_cells,
            verbose=True,
            num_of_processes=num_of_processes,
        )

        from microbiome._vendor.analysis_toolbox.utils import boolean_indexing

        self.all_cells_anat = boolean_indexing(self.all_cells_locs, fillval="Undefined")

        if persist:
            np.save(self.procpath + f"all_cells-anat.npy", self.all_cells_anat)

    def load_cell_anat(self, compute=False, **compute_kwargs):
        if os.path.exists(self.procpath + f"all_cells-anat.npy"):

            self.cell_labels = np.load(
                self.procpath + f"all_cells-anat.npy", allow_pickle=True
            )

        else:
            print(f"{self.procpath + f'all_cells-anat.npy'} doesn't exist")

            if compute:
                self.compute_cell_anat(**compute_kwargs)
            else:
                print(f"Not computing...")

    def segment_dual_color(self):
        if "ID" in self.seg_df.columns:
            self.neurons_idx = self.seg_df[self.seg_df["ID"] == "neuron"].index
        else:
            self.neurons_idx = np.arange(self.n)

        print(f"{len(self.neurons_idx)}/{self.n} segments are neurons")


class voluseg_regd(object):
    def __init__(self, voluseg, debug=False):
        """ """
        self.voluseg = voluseg
        self.debug = debug

    def load_reference_brain(
        self,
        reference_brain_path="/groups/ahrens/ahrenslab/jing/zebrafish_atlas/yumu_confocal/20150519/im/cy14_1p_stitched_xyz_z-flipped.nii.gz",
    ):

        from microbiome._vendor.analysis_toolbox.stacks_helper import load_defined_stack

        (
            volume_mean,
            volume_mean_shape,
            volume_mean_vox_spacing,
        ) = load_defined_stack(reference_brain_path)

        self.volume_mean = volume_mean.T
        self.volume_mean_shape = volume_mean_shape[::-1]
        self.vox_spacing = volume_mean_vox_spacing[::-1]

        self.z, self.y, self.x = self.volume_mean_shape

    def load_registered_seg_loc(self):

        ## TODO: used to be self.voluseg.segmented_root
        ## TODO: used to be self.procpath
        with np.load(self.voluseg.segmented_root + "registered_cell_locs.npz") as f:

            self.cell_x = f["registered_cell_x"]
            self.cell_y = f["registered_cell_y"]
            self.cell_z = f["registered_cell_z"]

            if "cell_weights" in f.files:
                self.cell_weights = f["cell_weights"]

    def find_cells(
        self,
        cell_list,
        cell_volume=None,
        mask=1,
        pixbypix=False,
        z_offset=0,
        y_offset=0,
        x_offset=0,
        vox_spacing=None,
        cell_size_um=None,
        debug=False,
    ):

        from microbiome._vendor.analysis_toolbox.data.voluseg import find_cells

        cell_volume = find_cells(
            cell_list,
            self.cell_z,
            self.cell_y,
            self.cell_x,
            self.voluseg.cell_weights,
            (self.z, self.y, self.x),
            cell_volume=cell_volume,
            mask=mask,
            pixbypix=pixbypix,
            z_offset=z_offset,
            y_offset=y_offset,
            x_offset=x_offset,
        )

        if vox_spacing is not None and cell_size_um is not None:

            from microbiome._vendor.analysis_toolbox.data.segments import dilate_cells_in_volume

            cell_volume = dilate_cells_in_volume(
                cell_volume, cell_size_um, vox_spacing, debug=debug
            )

        return cell_volume

    def make_projections(self, image_stack, background=False, contour=False, **kwargs):

        from microbiome._vendor.analysis_toolbox.stacks_helper import make_projections

        if background:
            background = self.volume_mean
        else:
            background = None

        if contour:
            contour = self.volume_mean
        else:
            contour = None

        return make_projections(
            image_stack,
            voxel_spacing=self.vox_spacing,
            background=background,
            contour=contour,
            **kwargs,
        )

    def setup_seg_df(self):

        if "cell_x" in self.voluseg.seg_df.columns:
            self.seg_df = self.voluseg.seg_df.drop(
                columns=["cell_x", "cell_y", "cell_z"]
            )
        elif "weights" in self.voluseg.seg_df.columns:
            self.seg_df = self.voluseg.seg_df.drop(columns=["weights", "origins"])
            self.seg_df["cell_weights"] = pd.Series(list(getattr(self, "cell_weights")))

        for dim in ["x", "y", "z"]:

            self.seg_df[f"cell_{dim}"] = pd.Series(list(getattr(self, f"cell_{dim}")))

    def find_segment_centroids(self):

        new_cols = ["Centroid x", "Centroid y", "Centroid z"]

        self.seg_df[new_cols] = self.seg_df.progress_apply(
            find_segment_centroids, axis=1
        )


def return_cells_in_region(all_cells_anat, regions, return_mode="indices"):
    """
    all_cells_anat [np.array of size ncells x 8]: all regions that cell resides in
    regions [list]: list of regions
    """
    from microbiome._vendor.analysis_toolbox.utils import operate_on_first_dim

    cell_mask = np.logical_or.reduce(
        [operate_on_first_dim(all_cells_anat, r) for r in regions]
    )

    if return_mode == "indices":
        return np.where(cell_mask)[0]
    elif return_mode == "boolean":
        return cell_mask


class cellProcessing(voxel):
    """Load output files generated by fish_processing/fish_proc/wholeBrainDask/cellProcessing_single_WS.py"""

    def __init__(
        self,
        segmented_root="",
        im_dir="",
        external_segmented_root="",
        # load options
        mode="simple",
        # load_filtered
        # load_unfiltered
        # load_filtered_dff
        # load_segment_loc
        # extra options
        metadata="",
        debug=False,
    ):

        ## Sets imaging related variables #################
        # self.path, self.debug, self.metadata_dir,
        # self.savepath, self.procpath,
        # self.expt_id, self.expt_date, self.expt_name,
        # self.metadata, self.metadata_pd
        ###################################################
        # if segmented is not a subdir of its original directory
        # (e.g. when it is in Ziqiang's folder), we still want to save
        # analysis/results etc in the original directory
        if im_dir:

            super().__init__(im_dir=im_dir, metadata=metadata, debug=debug)

        # if segmented is in a subdir of its original directory,
        # we can use this directory to find the right subdirs to save
        # analysis/results etc
        else:

            super().__init__(im_dir=segmented_root, metadata=metadata, debug=debug)

        self.segmented_root = segmented_root
        self.external_segmented_root = external_segmented_root
        self.debug = debug
        self.metadata_dir = metadata

        from microbiome._vendor.analysis_toolbox.utils import path_slash

        self.brain_file = path_slash(segmented_root) + "Y_ave.npy"
        self.filtered_segments = path_slash(segmented_root) + "cell_raw_dff_sparse.npz"
        self.unfiltered_segments = path_slash(segmented_root) + "cell_raw_dff.npz"
        self.filtered_dff_segments = path_slash(segmented_root) + "cell_dff.npz"

        self.mode = mode

        # General mixing up of filtered and unfiltered cells
        # TODO: only load the filtered cells!
        self.load_brain_vol()  # always load the mean file

        if self.mode == "load_filtered" or self.mode == "full":
            self.load_filtered_segments()

        if self.mode == "load_unfiltered" or self.mode == "full":
            self.load_unfiltered_segments()

        if (
            self.mode == "load_filtered_dff"
            or self.mode == "simple"
            or self.mode == "full"
        ):
            self.load_filtered_dff_segments()

        if (self.mode == "load_segment_loc" or self.mode == "loc") and not np.any(
            [
                self.mode == "load_filtered",
                self.mode == "load_unfiltered",
                self.mode == "load_filtered_dff",
            ]
        ):
            self.load_segment_locations()
            self.load_brain_vol()

        if self.mode == "nothing":
            pass

        self.compute_simple_variables()

    ########################
    #       Data I/O       #
    ########################

    def load_brain_vol(self):

        self.volume_mean = np.squeeze(np.load(self.brain_file))
        self.z, self.y, self.x = self.volume_mean.shape

    def load_segment_locations(self):
        """
        Load the dFF of cells after cells with bad SNR have been
        filtered out. DFF calculation is done on `sparse` segments,
        which mean that cells outside the brain have already been
        removed.

        Ziqiang: Final data is cell-dff, which has dff and some filter on dff
        keys: ['A_loc', 'A', 'dFF']
        """

        self.cmpnts = np.load(self.filtered_dff_segments, allow_pickle=True)

        self.cmpnt_wghts = self.cmpnts["A"]
        self.cmpnt_orig = self.cmpnts["A_loc"]

        print("Segment locations loaded!")

    def load_filtered_dff_segments(self):
        """
        Load the dFF of cells after cells with bad SNR have been
        filtered out. DFF calculation is done on `sparse` segments,
        which mean that cells outside the brain have already been
        removed.

        Ziqiang: Final data is cell-dff, which has dff and some filter on dff
        keys: ['A_loc', 'A', 'dFF']
        """

        self.cmpnts = np.load(self.filtered_dff_segments, allow_pickle=True)

        self.cmpnt_wghts = self.cmpnts["A"]
        self.cmpnt_dff = self.cmpnts["dFF"]
        self.cmpnt_orig = self.cmpnts["A_loc"]

        print("Filtered components loaded!")

    def load_filtered_segments(self):
        """
        Cells that are outside of the brain are removed.

        Ziqiang: Raw-sparse is the data with spatial filter.
        keys: ['A_loc', 'A_shape', 'A', 'F']
        """

        self.fCmpnts = np.load(self.filtered_segments, allow_pickle=True)

        self.fCmpnt_wghts = self.fCmpnts["A"]
        self.fCmpnt_F = self.fCmpnts["F"]
        self.fCmpnt_orig = self.fCmpnts["A_loc"]

        print("Filtered components loaded!")

    def load_unfiltered_segments(self):
        """
        Load all cells detected by the segmentation algorithm.
        Ziqiang: Raws are f data
        keys: ['A_loc', 'A_shape', 'A', 'F']
        """

        self.Cmpnts = np.load(self.unfiltered_segments, allow_pickle=True)

        self.Cmpnt_wghts = self.Cmpnts["A"]
        self.Cmpnt_F = self.Cmpnts["F"]
        self.Cmpnt_orig = self.Cmpnts["A_loc"]

        print("Unfiltered components loaded!")

    def compute_simple_variables(self):
        # compute commonly used variables
        if hasattr(self, "cmpnt_dff"):
            self.n = self.cmpnt_dff.shape[0]
            self.t = self.cmpnt_dff.shape[1]
        if hasattr(self, "cmpnt_wghts"):
            self.n = len(self.cmpnt_wghts)
        if hasattr(self, "z") and hasattr(self, "y") and hasattr(self, "x"):
            self.dset_dims = np.array([self.z, self.y, self.x])

    ########################
    #    Pre-processing    #
    ########################
    def put_segments_in_df(self):

        if hasattr(self, "cmpnt_wghts"):
            seg_dict = {
                "weights": list(self.cmpnt_wghts),
                "origins": list(self.cmpnt_orig),
            }

            if hasattr(self, "cmpnt_dff"):
                seg_dict["dff"] = list(self.cmpnt_dff)

            setattr(self, "seg_df", pd.DataFrame(seg_dict))

    def find_z(self, seg):
        return seg["origins"][0]  # z

    def find_segment_centroids(self):

        # calculate centroids
        from scipy.ndimage.measurements import center_of_mass

        self.seg_df["Big weights"] = self.seg_df["weights"].progress_apply(
            lambda x: x > (np.nanmax(x) * 0.4)
        )
        self.seg_df["Center of mass"] = self.seg_df["Big weights"].progress_apply(
            lambda x: center_of_mass(x)
        )

        self.seg_df["Centroid x"] = self.seg_df["Center of mass"].apply(
            lambda x: x[1]
        ) + self.seg_df["origins"].apply(lambda x: x[2])
        self.seg_df["Centroid y"] = self.seg_df["Center of mass"].apply(
            lambda x: x[0]
        ) + self.seg_df["origins"].apply(lambda x: x[1])
        self.seg_df["Centroid z"] = self.seg_df["origins"].apply(lambda x: x[0])

    def filter_cells(self, max_thresh=None, min_thresh=None, confirm=False):
        """
        Only include cells that have activity that are greater than 20
        F (to cells with only "noise") and with non-negative activity
        (which could mean that it is at the edge of the brain). Cells
        with some levels of activity that are at the edge of the brain
        will also be removed.
        """
        if max_thresh is not None:
            cmpnts_max = self.Cmpnt_F.max(axis=-1).compute() > max_thresh
        if min_thresh is not None:
            cmpnts_min = self.Cmpnt_F.min(axis=-1).compute() > min_thresh

        if max_thresh is not None and min_thresh is not None:
            cmpnts = np.logical_or(cmpnts_max, cmpnts_min)
        elif max_thresh is None:
            cmpnts = cmpnts_min
        elif min_thresh is None:
            cmpnts = cmpnts_max

        # NOTE: the original is not replaced
        cmpnt_wghts = self.Cmpnt_wghts[cmpnts]
        cmpnt_F = self.Cmpnt_F[cmpnts]
        cmpnt_orig = self.Cmpnt_orig[cmpnts]

        accept = None

        if confirm:
            self.weiz_visualize_all_cells(cmpnt_wghts.compute(), cmpnt_orig.compute())
            accept = input("Accept? (Y/N)")

        if accept == "Y" or accept or ~confirm:
            self.cmpnt_wghts = cmpnt_wghts
            self.cmpnt_F = cmpnt_F
            self.cmpnt_orig = cmpnt_orig

        else:
            del cmpnt_wghts
            del cmpnt_F
            del cmpnt_orig

    def calculate_DFF(
        self,
        window=400,
        percentile=20,
        downsample=10,
        baseline_offset=10,
        lazy=False,
        persist=True,
        debug=False,
    ):

        import fish_proc.wholeBrainDask.cellProcessing_single_WS as fwc

        cmpnt_F = self.cmpnt_F.rechunk(chunks=("auto", -1))
        print(cmpnt_F)

        self.baseline = da.map_blocks(
            fwc.baseline,
            cmpnt_F,
            dtype="float",
            window=window,
            percentile=percentile,
            downsample=downsample,
        )

        self.cmpnt_dff = (self.cmpnt_F - self.baseline) / (
            self.baseline + baseline_offset
        )

        if not lazy:
            if persist:
                da.persist(self.cmpnt_F, self.baseline, self.cmpnt_dff)
            else:
                da.compute(self.cmpnt_F, self.baseline, self.cmpnt_dff)

        if debug:
            # Possible issues
            # 1: division by zero
            # 2: signal too small due to high camera offset?

            cell_mean = da.nanmean(self.cmpnt_dff, axis=1).compute()

            plt.figure()
            plt.plot(cell_mean)
            plt.set_ylabel(r"Mean $\delta{}F/F$ across time")
            plt.set_xlabel(r"Cell #")

    ########################
    #    Visualization     #
    ########################

    def weiz_visualize_all_cells(self, weights, origins):

        wei_visualize_all_cells(weights, origins, self.volume_mean)

    def find_cell(
        self,
        cell_num,
        cell_volume=None,
        mask=1,
        pixbypix=False,
        z_offset=0,
        y_offset=0,
        x_offset=0,
    ):

        from microbiome._vendor.analysis_toolbox.cell_helper import find_cell

        return find_cell(
            cell_num,
            self.cmpnt_orig,
            self.cmpnt_wghts,
            self.volume_mean[z_offset:, y_offset:, x_offset:],
            cell_volume=cell_volume,
            mask=mask,
            pixbypix=pixbypix,
            z_offset=z_offset,
            y_offset=y_offset,
            x_offset=x_offset,
        )

    def normalize_component_weights(self):

        row_wghts = np.nanmax(self.cmpnt_wghts, axis=(1, 2), keepdims=True)
        norm_wghts = self.cmpnt_wghts / row_wghts

        self.norm_cmpnt_wghts = norm_wghts

    def find_cells(
        self,
        cell_list,
        cell_volume=None,
        mask=False,
        z_offset=0,
        y_offset=0,
        x_offset=0,
        weight_multiplier=1,
    ):

        if not hasattr(self, "norm_cmpnt_wghts"):
            self.normalize_component_weights()

        from microbiome._vendor.analysis_toolbox.cell_helper import find_cells

        return find_cells(
            cell_list,
            self.cmpnt_orig,
            self.norm_cmpnt_wghts * weight_multiplier,
            self.volume_mean[z_offset:, y_offset:, x_offset:],
            cell_volume=cell_volume,
            mask=mask,
            z_offset=z_offset,
            y_offset=y_offset,
            x_offset=x_offset,
        )

    def initialize_regd(self, force=None, debug=False):
        """
        TODO: REPEATED
        """

        if (
            os.path.exists(self.procpath + "registered_cell_locs.npz")
            and force != "weiz"
        ):

            print("Found processed files for Javier's registration...")

            self.regd = voluseg_regd(self)

        elif os.path.exists(self.external_segmented_root) and force != "howj":

            print("Found processed files in Ziqiang's directory")

            from statemod.givingup_helper import statemod_cellProcessing_regd

            self.regd = statemod_cellProcessing_regd(self.external_segmented_root)
            self.regd.load_all_cell_centers(debug=debug)
            self.regd.index_brain_cell_centers()
            self.regd.load_atlas_labels()

            ## self.regd = cellProcessing_regd(self.segmented_root) also works

    def load_cell_anat(self, compute=False, **compute_kwargs):
        if os.path.exists(self.procpath + f"all_cells-anat.npy"):

            self.cell_labels = np.load(
                self.procpath + f"all_cells-anat.npy", allow_pickle=True
            )

        else:
            print(f"{self.procpath + f'all_cells-anat.npy'} doesn't exist")

            if compute:
                self.compute_cell_anat(**compute_kwargs)
            else:
                print(f"Not computing...")

    def segment_dual_color(self):
        if "ID" in self.seg_df.columns:
            self.neurons_idx = self.seg_df[self.seg_df["ID"] == "neuron"].index
        else:
            self.neurons_idx = np.arange(self.n)

        print(f"{len(self.neurons_idx)}/{self.n} segments are neurons")

    def compute_cell_anat(self, num_of_processes=None, persist=False):
        """
        TODO: Repeated in voluseg
        """

        if not hasattr(self, "regd"):

            self.initialize_regd()

        from statemod.givingup_helper import statemod_cellProcessing_regd

        if isinstance(self.regd, voluseg_regd):

            self.regd.load_registered_seg_loc()
            self.regd.load_reference_brain()
            self.regd.setup_seg_df()
            self.regd.find_segment_centroids()
            self.all_cells_zyx = self.regd.seg_df[
                ["Centroid z", "Centroid y", "Centroid x"]
            ].to_numpy()

        elif isinstance(self.regd, statemod_cellProcessing_regd):

            if hasattr(self.regd, "registered_cell_centers_raw_space"):
                self.all_cells_zyx = self.regd.registered_cell_centers_raw_space
            else:
                self.all_cells_zyx = self.regd.registered_cell_centers

        print(f"Computing for {self.expt_id}...")

        from microbiome._vendor.analysis_toolbox.parallel import parallelize, num_cpu
        from microbiome._vendor.analysis_toolbox.data.segments import (
            compute_cell_location_for_multiple_cells,
        )

        if num_of_processes is None:
            num_of_processes = num_cpu()

        self.all_cells_locs = parallelize(
            self.all_cells_zyx.astype("int"),
            compute_cell_location_for_multiple_cells,
            verbose=True,
            num_of_processes=num_of_processes,
        )

        from microbiome._vendor.analysis_toolbox.utils import boolean_indexing

        self.all_cells_anat = boolean_indexing(self.all_cells_locs, fillval="Undefined")

        if persist:
            np.save(self.procpath + f"all_cells-anat.npy", self.all_cells_anat)


def dilate_cells_in_volume(cell_volume, cell_size_um, vox_spacing, debug=False):

    from microbiome._vendor.analysis_toolbox.utils import solve_ratio

    isometric_radius = solve_ratio(1 / vox_spacing)
    print(f"Isometric radius: {isometric_radius}")

    import raster_geometry as rg

    struct = rg.ellipsoid(
        int(np.max(np.array(isometric_radius) * 2) + 3), isometric_radius
    )
    # add 3 because one for center, and two for padded sides

    isometric_diameter = vox_spacing * np.array(isometric_radius) * 2
    from microbiome._vendor.analysis_toolbox.utils import check_binary, check_all_elements_equal

    print(f"Isometric diameter: {isometric_diameter}")
    # assert check_all_elements_equal(isometric_diameter)

    iterations = np.round(cell_size_um / isometric_diameter[0]).astype("int")
    print(f"Dilation iterations: {iterations}")

    if iterations > 0:
        if check_binary(cell_volume):
            print("binary")

            from scipy.ndimage.morphology import binary_dilation

            cell_volume = binary_dilation(
                cell_volume, structure=struct, iterations=iterations
            ).astype("float")

        else:
            print("non-binary")
            cell_volume = nonbinary_dilation(
                cell_volume, structure=struct, iterations=iterations
            )

    if debug:
        plt.figure()
        plt.imshow(np.nanmax(cell_volume, axis=0))

    return cell_volume


def nonbinary_dilation(A, structure, iterations):

    if np.nanmax(A) > 0 and np.nanmin(A) >= 0:
        filter_funcs = ["max"]
    elif np.nanmax(A) <= 0 and np.nanmin(A) < 0:
        filter_funcs = ["min"]
    else:
        filter_funcs = ["min", "max"]

    for filter_func in filter_funcs:
        for i in tqdm(range(iterations)):
            A = nonbinary_dilate(A, structure, filter_func=filter_func)

    return A


def nonbinary_dilate(A, structure, filter_func="max"):

    # see the following for full list:
    # https://scipy.github.io/devdocs/tutorial/ndimage.html#filters-based-on-order-statistics
    from scipy.ndimage import maximum_filter, minimum_filter, median_filter

    if isinstance(filter_func, str):
        if filter_func == "max":
            filter_func = maximum_filter
        elif filter_func == "min":
            filter_func = minimum_filter
        elif filter_func == "med":
            filter_func = median_filter

    print(f"Applying {filter_func}...")
    B = filter_func(A, footprint=structure)
    B[A != 0] = A[A != 0]

    return B


def load_atlas_labels(
    atlas_label_mat_path="/groups/ahrens/ahrenslab/jing/zebrafish_atlas/yumu_confocal/20150519/processed/atlas_label_mat.npz",
):
    """
    Takes about 7s to load.

    atlas_label_mat (shape: (170, 1325, 2509)):
        total number of voxels: np.multiply.reduce(atlas_label_mat.shape) = 565152250
        number of labelled voxels: np.sum(atlas_label_mat >= 0) = 557334588
        Example: number of voxels with label 14033092: np.sum(atlas_label_mat == 14033092)
    mask_mat (shape: (120491388, 294)): all unique label combinations and their labels in boolean form
    mask_labels (shape: (294,)): tuple of all 294 labels
    """

    from scipy import sparse

    _ = np.load(atlas_label_mat_path, allow_pickle=True)
    atlas_label_mat = _["atlas_label_mat"]
    mask_mat = sparse.csc_matrix(
        (_["mask_data"], _["mask_ir"], _["mask_jc"]), shape=_["shape"]
    )
    mask_labels = _["mask_labels"]

    return atlas_label_mat, mask_mat, mask_labels


def compute_cell_location(zyx_loc, mask_mat, atlas_label_mat, mask_labels):

    z, y, x = zyx_loc

    from microbiome._vendor.analysis_toolbox.utils import isnan

    if (not isnan(x)) and (not isnan(y)) and (not isnan(x)):

        if np.all(np.array((z, y, x)) < np.array(atlas_label_mat.shape)) and np.all(
            np.array((z, y, x)) >= 0
        ):

            atlas_zbrain = mask_mat[atlas_label_mat[z, y, x]].toarray()[0]
            labels = mask_labels[np.where(atlas_zbrain)[0]]

            return labels
        else:
            return np.array((np.nan,))

    else:
        return np.array((np.nan,))


def compute_cell_location_for_multiple_cells(
    zyx_locs,
    atlas_label_mat_path="/nrs/ahrens/Ziqiang/Atlas/ZBrain/atlas_label_mat.npz",
):

    atlas_label_mat, mask_mat, mask_labels = load_atlas_labels(
        atlas_label_mat_path=atlas_label_mat_path
    )

    return [
        compute_cell_location(zyx_loc, mask_mat, atlas_label_mat, mask_labels)
        for zyx_loc in zyx_locs
    ]


def find_segment_centroids(series):

    cell_weights = series["cell_weights"][np.isfinite(series["cell_weights"])]
    cell_weights = cell_weights[cell_weights >= 0]

    cell_x_col = "cell_x"
    cell_y_col = "cell_y"
    cell_z_col = "cell_z"

    cell_x = series[cell_x_col][series[cell_x_col] >= 0]
    cell_y = series[cell_y_col][series[cell_y_col] >= 0]
    cell_z = series[cell_z_col][series[cell_z_col] >= 0]

    # number of cell_weights and cell_x coming out of mika's pipeline should
    # be the same. However, some of those elements might be removed after alignment
    # because it is outside of the space of the reference brain
    if cell_weights.shape != cell_x.shape:
        cell_weights_x = series["cell_weights"][series[cell_x_col] >= 0]
    else:
        cell_weights_x = cell_weights

    if cell_weights.shape != cell_y.shape:
        cell_weights_y = series["cell_weights"][series[cell_y_col] >= 0]
    else:
        cell_weights_y = cell_weights

    if cell_weights.shape != cell_z.shape:
        cell_weights_z = series["cell_weights"][series[cell_z_col] >= 0]
    else:
        cell_weights_z = cell_weights

    centroid_x = (cell_weights_x @ cell_x) / np.nansum(cell_weights_x)
    centroid_y = (cell_weights_y @ cell_y) / np.nansum(cell_weights_y)

    unique_cell_z = np.unique(cell_z)

    if len(unique_cell_z) == 1:
        centroid_z = unique_cell_z[0]
    else:
        centroid_z = (cell_weights_z @ cell_z) / np.nansum(cell_weights_z)

    return pd.Series([centroid_x, centroid_y, centroid_z])
