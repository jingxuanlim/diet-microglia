# Vendored from analysis_toolbox (data/imaging.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np
import pandas as pd
from microbiome._vendor.analysis_toolbox.data.optics import optics


class voxel(optics):

    def __init__(self,
                 im_dir,
                 metadata='',
                 debug=False):

        ######################
        # imaging attributes #
        ######################

        ## set path, ephys_file and debug
        super().__init__(im_dir=im_dir,
                         metadata=metadata,
                         debug=debug)

    def compute_vox_spacing(self):

        self.vox_spacing = (self.raw_dims / self.dset_dims) * self.raw_vox_spacing

    def compute_image_stats(self):

        self.stack_shape = self.data.shape[1:]
        self.dset_dims = self.stack_shape   ## to be consistent with segments
        self.nvoxels = np.multiply.reduce(self.stack_shape)
        self.ntimepts = self.data.shape[0]

    def put_voxels_in_df(self):

        z_idx, y_idx, x_idx = np.where(np.ones(self.data.shape[1:]))

        voxel_dict = {
            'z': list(z_idx),
            'y': list(y_idx),
            'x': list(x_idx)
        }

        setattr(self, 'voxel_df', pd.DataFrame(voxel_dict))
