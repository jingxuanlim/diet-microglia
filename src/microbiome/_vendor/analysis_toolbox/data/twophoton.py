# Vendored from analysis_toolbox (data/twophoton.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


from microbiome._vendor.analysis_toolbox.data.imaging import voxel


class twophoton(voxel):
    """Same level as ephys class"""

    def __init__(self, im_dir, metadata="", debug=False):

        ######################
        # imaging attributes #
        ######################

        ## set path, ephys_file and debug
        super().__init__(im_dir=im_dir, metadata=metadata, debug=debug)

        self.load_images()
        self.compute_image_stats()

    def load_images(self):

        import os
        from microbiome._vendor.analysis_toolbox.utils import ensure_not_end_in_slash

        filename, fileext = os.path.splitext(ensure_not_end_in_slash(self.path))

        if fileext == ".tif":
            self.load_tif_images()
        elif fileext == ".zarr":
            self.open_images()
        elif fileext == ".nd2":
            self.load_nd2_images()

    def open_images(self, chunks=None):
        """Open images lazily with dask

        PIPELINE (3)
        A wrapper for Davis' code to open the raw light sheet imaging
        data.
        """

        from microbiome._vendor.analysis_toolbox.dataorg.dataset import ensure_array

        self.data = ensure_array(self.path, chunks=chunks)

    def load_zarr_images(self):

        import zarr

        self.data = zarr.load(self.path)

    def read_zarr_images(self):

        import zarr

        self.data = zarr.open(self.path, mode="r")

    def load_tif_images(self):

        from skimage.io import imread

        self.data = imread(self.path)

    def load_nd2_images(self):

        import nd2

        self.data = nd2.imread(self.path, xarray=True, dask=True)
        self.metadata = nd2.ND2File(self.path)

    def preprocess(self):

        import numpy as np

        data_min = np.nanmin(self.data)
        print(f"The smallest value in the raw data is: {data_min}")

        self.data = self.data - data_min
