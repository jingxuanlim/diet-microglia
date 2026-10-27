# Vendored from analysis_toolbox (data/optics.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np


class optics(object):
    """
    Principal class for all datasets obtained using optical methods.
    For a given path of the imaging dataset and its corresponding ephys
    file, infer properties of the experiment and define important directories.

    Made with light sheet imaging dataset, but with also other imaging
    data types in mind (e.g. 2P).
    """

    def __init__(self,
                 im_dir,
                 metadata='',
                 debug=False):

        self.path = im_dir
        self.debug = debug
        self.metadata_dir = metadata

        # where all figures should go to
        self.savepath = '/'.join(self.path.split('/')[:-2]) + '/analysis/'

        # where the data, in its various levels of processing goes to
        self.procpath = '/'.join(self.path.split('/')[:-2]) + '/processed/'

        # from microbiome._vendor.analysis_toolbox.utils import search_list_of_str
        # from microbiome._vendor.analysis_toolbox.dataorg.dataset import date_regex, expt_id_regex
        # # the unique identifier of any experiment
        # self.expt_id = search_list_of_str(self.path.split('/'), expt_id_regex)[0]

        # # when the experiment was done
        # from datetime import datetime
        # self.expt_date = datetime.strptime(search_list_of_str(self.path.split('/'),
        #                                                       date_regex)[0],
        #                                    '%Y%m%d')

        # # name of the experiment, which should contain information
        # # about the age of the fish, the genetic background of the
        # # fish, the experimental paradigm, etc
        # self.expt_name = '_'.join(self.expt_id.split('_')[:-2])
        from microbiome._vendor.analysis_toolbox.dataorg.dataset import identify_dataset
        self.expt_id,\
        self.expt_date,\
        self.expt_name,\
        self.fish_num,\
        self.expt_num =\
            identify_dataset(self.path)

        ## load metadata
        ## setting of voxel spacing shouldn't be here
        if self.metadata_dir:

            from microbiome._vendor.analysis_toolbox.dataorg.dataset import load_metadata
            self.metadata, self.metadata_pd = load_metadata(self.metadata_dir)

            if self.metadata_pd is not None:
                if 'z_step' in self.metadata_pd.index:
                    self.raw_vox_spacing = \
                        np.array([self.metadata_pd.loc['z_step', 'value'], 0.40625, 0.40625]).astype('float')  # zyx
                    self.raw_dims = \
                        np.array(self.metadata_pd.loc['dimensions', 'value'].split(';')[0].split('x')).astype('int')[::-1]

        else:

            self.metadata = None
            self.metadata_pd = None
