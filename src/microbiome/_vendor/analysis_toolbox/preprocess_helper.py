# Vendored from analysis_toolbox (preprocess_helper.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np


def binarize_swimming(swim_starts, swim_stops):

    # find periods with swims
    swim_start_idx = np.where(swim_starts)[0]
    swim_stop_idx = np.where(swim_stops)[0]

    swimming = np.zeros(swim_starts.shape)

    for start, stop in zip(list(swim_start_idx), list(swim_stop_idx)):
        swimming[start:stop] = 1.0

    return swimming, swim_start_idx, swim_stop_idx
