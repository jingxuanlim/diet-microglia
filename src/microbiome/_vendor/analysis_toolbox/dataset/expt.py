# Vendored from analysis_toolbox (dataset/expt.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


from pprint import pprint
import os


class expt(object):

    def __init__(
        self,
        # Experimental components
        spim_ops=None,
        twophoton_ops=None,
        ephys_ops=None,
        seg_ops=None,
        behavior_ops=None,
        metadata_dir="",
        debug=False,
    ):
        """Instructions for the loading of data that belong to experiments

        Experiments are defined based on its components

        Every experiment is different and is made up by one or more
        components. Each component (e.g. spim (imaging), ephys and
        behavior) indicated will be treated in a way described by the
        methods of this class.

        Each kwargs should represent a new component of an experiment
        (e.g. tail-free images). Increase the number of kwargs and
        methods as experiments evolve and become more complex.

        As you increase the number of experimental components, it
        makes sense to also write methods for them. These methods
        are typically are typically used to load the data.

        To be consistent, each kwarg should be a dictionary of
        arguments.
        """

        # Experimental components
        self.spim_ops = spim_ops
        self.twophoton_ops = twophoton_ops
        self.ephys_ops = ephys_ops
        self.behavior_ops = behavior_ops  # dummy: not used yet
        self.seg_ops = seg_ops

        # Experimental/Analysis properties
        self.metadata_dir = metadata_dir
        self.root_dir = metadata_dir
        self.debug = debug

        # Load metadata
        self.load_metadata()

        # all figures should go here
        self.savepath = self.root_dir + "analysis/"
        self.procpath = self.root_dir + "processed/"

        # TODO: define experiment name etc for ID-ing

    def load_metadata(self):

        from microbiome._vendor.analysis_toolbox.dataorg.dataset import load_metadata

        self.metadata, self.metadata_pd = load_metadata(self.metadata_dir)

    def load_ephys(self):
        """Load electrophysiology data

        Takes in either .10chFlt or .npy (processed) ephys data.

        Returns
        --------
        ephys: ephys object

        """

        if "ephys_file" in self.ephys_ops:

            from microbiome._vendor.analysis_toolbox.data.ephys import ephys

            self.ep = ephys(
                ephys_file=self.ephys_ops["ephys_file"],
                channel_labels=self.ephys_ops["channel_labels"],
                metadata=self.ephys_ops["metadata"],
                debug=self.debug,
            )

    def load_twophoton(self):
        """Load two-photo imaging data"""

        if "im_path" in self.twophoton_ops:

            print("Raw images requested...")

            from microbiome._vendor.analysis_toolbox.data.twophoton import twophoton

            for im_dir in self.twophoton_ops["im_path"]:

                im_name = im_dir.split("/")[-2] + "_raw"

                setattr(
                    self,
                    im_name,
                    twophoton(
                        im_dir=im_dir,
                        metadata=self.twophoton_ops["metadata"],
                        debug=self.debug,
                    ),
                )

                print(hasattr(self, im_name))

        if "impro_dir" in self.twophoton_ops:

            print("Processed iamges requested...")

            from microbiome._vendor.analysis_toolbox.data.twophoton import twophoton

            for impro_dir in self.twophoton_ops["impro_dir"]:

                print(impro_dir)
                file_name = impro_dir.split("/")[-2]
                print(file_name)
                filename, fileext = os.path.splitext(file_name)
                im_name = filename
                print(f"Setting attribute {im_name}")

                setattr(
                    self,
                    im_name,
                    twophoton(
                        im_dir=impro_dir,
                        metadata=self.twophoton_ops["metadata"],
                        debug=self.debug,
                    ),
                )

    def load_spim(self):
        """Load light sheet imaging data

        Currently takes in both images and segments.
        # TODO: split spim up into raw spim images, processed spim
        # images and segmented cells as they either one of them
        # constitute a complete imaging dataset that can be used for
        # analysis.

        Returns
        --------
        spim: spim or segmented object
        segmented_bool [boolean]: whether segmented object exist

        """
        if "im_dir" in self.spim_ops:

            print("Raw images requested...")

            from microbiome._vendor.analysis_toolbox.data.spim import spim_raw

            for im_dir in self.spim_ops["im_dir"]:

                if im_dir == "im":
                    # there's only one imaging directory, so
                    # let's name it camera CM0
                    im_name = "im_CM0"

                im_name = im_dir.split("/")[-2] + "_raw"

                if os.path.exists(im_dir + "TM0000000_CM0_CHN00.h5") or os.path.exists(
                    im_dir + "TM0000000_CM1_CHN00.h5"
                ):
                    setattr(
                        self,
                        im_name,
                        spim_raw(
                            im_dir=im_dir,
                            metadata=self.spim_ops["metadata"],
                            debug=self.debug,
                        ),
                    )

                if os.listdir(im_dir)[0].endswith(".TIF") or os.listdir(im_dir)[0].endswith(".tif"):
                    setattr(
                        self,
                        im_name,
                        spim_raw(
                            im_dir=im_dir,
                            metadata=self.spim_ops["metadata"],
                            debug=self.debug,
                        ),
                    )
                else:
                    print(
                        f"Error loading raw images for {im_name}. TM0000000_CMX_CHN00.h5 not found!"
                    )

        # TODO: scale this up to multiple datasets
        if "impro_dir" in self.spim_ops:

            print("Processed images requested...")

            from types import SimpleNamespace
            import copy
            from microbiome._vendor.analysis_toolbox.data.spim import spim_proc

            # prepare class for all processed data
            self.proc_spim = copy.deepcopy(self.spim_ops["impro_dir"])

            for proc, proc_path in self.proc_spim.items():
                print(f"Attempting to load {proc_path}")

                setattr(
                    self,
                    proc,
                    spim_proc(
                        im_dir=proc_path,
                        metadata=self.spim_ops["metadata"],
                        debug=self.debug,
                    ),
                )

        if "segmented_root" in self.spim_ops:

            if self.debug:
                print("parsing segmented data...")
                pprint(self.spim_ops["segmented_root"])

            for segmented_root, segmented_im_dir, external_segmented_root in zip(
                self.spim_ops["segmented_root"],
                self.spim_ops["segmented_im_dir"],
                self.spim_ops["external_segmented_root"],
            ):

                # what to name the segmented attribute?
                # won't work when segmented is in ziqiang's directory, but this is
                # fixed in the next step with standard name im_CM0
                segmented_name = segmented_root.split("/")[-2]

                print(f"segmented_root: {segmented_root}")

                if ("_seg" not in segmented_name) and (
                    "_voluseg" not in segmented_name
                ):

                    # rename certain datasets so that experiment has
                    # either im_CM0 or im_CM0 & im_CM1
                    if not any(
                        subdir in segmented_name
                        for subdir in ["im", "im_CM0", "im_CM1"]
                    ):
                        # segmented dir must not have been in those subdirectories
                        # there's likely only one imaging datase
                        # let's name it im_CM0
                        segmented_name = "im_CM0"

                    elif segmented_name == "im":
                        # again, there's only one imaging directory, so
                        # let's name it camera CM0
                        segmented_name = "im_CM0"

                    segmented_name += "_seg"

                print(f"segmented_name: {segmented_name}")
                from microbiome._vendor.analysis_toolbox.data.segments import voluseg, cellProcessing

                if "_voluseg" in segmented_name:
                    setattr(
                        self,
                        segmented_name,
                        voluseg(
                            segmented_root=segmented_root,
                            im_dir=segmented_im_dir,
                            external_segmented_root=external_segmented_root,
                            mode=self.spim_ops["mode"],
                            metadata=self.spim_ops["metadata"],
                            debug=self.debug,
                        ),
                    )

                elif "_seg" in segmented_name:
                    setattr(
                        self,
                        segmented_name,
                        cellProcessing(
                            segmented_root=segmented_root,
                            im_dir=segmented_im_dir,
                            external_segmented_root=external_segmented_root,
                            mode=self.spim_ops["mode"],
                            metadata=self.spim_ops["metadata"],
                            debug=self.debug,
                        ),
                    )

                else:
                    print("Segmented directory not recognised!")

        if "reference_stack" in self.spim_ops:

            from microbiome._vendor.analysis_toolbox.data.spim import spim

            self.im_ref = spim(
                im_dir=self.spim_ops["reference_stack"],
                metadata=self.spim_ops["metadata"],
                debug=self.debug,
            )
