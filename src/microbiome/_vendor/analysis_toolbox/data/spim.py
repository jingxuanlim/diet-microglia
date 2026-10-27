# Vendored from analysis_toolbox (data/spim.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import os
from microbiome._vendor.analysis_toolbox.data.imaging import voxel


class spim(voxel):

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

    def open_images(self, chunks=None, debug=True):
        """Open images lazily with dask

        PIPELINE (3)
        A wrapper for Davis' code to open the raw light sheet imaging
        data.
        """

        from microbiome._vendor.analysis_toolbox.dataorg.dataset import ensure_array

        if debug:
            print(f"Opening {self.path}")
        self.data = ensure_array(self.path, chunks=chunks)


class spim_raw(spim):

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

        ###################
        # spim attributes #
        ###################

        ## TODO: for some reason, try didn't throw an error where there was
        # a problem using ZDS() within that open_raw_images() method...

        self.open_images(chunks=(1,-1,-1,-1)) ## used to be chunks=10 but not sure why
        self.compute_image_stats()

        print('Raw images loaded!\n')


class spim_proc(spim):

    """Same level as ephys class
    """
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
        
        ###################
        # spim attributes #
        ###################

        self.identify_dataset()
        self.open_images(chunks=None)
        self.compute_image_stats()
        self.load_volume_mean()

        print(f"Processed imaging data {self.path} loaded!")

    def identify_dataset(self):

        from microbiome._vendor.analysis_toolbox.utils import basename

        bname = basename(self.path)

        imname = bname.split('.')[0]  ## remove extensions

        components = imname.split('_')

        if len(components) == 4:
            self.datatype, self.channel, self.processedtype, self.chunktype = components
        elif len(components) == 3:
            self.datatype, self.channel, self.processedtype = components
        elif len(components) == 2:
            self.datatype, self.channel = components

        self.im_mode = '_'.join([self.datatype, self.channel, self.processedtype])


    def load_volume_mean(self):

        from microbiome._vendor.analysis_toolbox.fileio import readfrom_h5

        volume_mean_path = self.procpath + f'{self.im_mode}_background_stack.h5'

        if os.path.exists(volume_mean_path):
            setattr(self,
                    'volume_mean',
                    readfrom_h5(volume_mean_path, dset_names=['ref'])['ref'])
