# Vendored from analysis_toolbox (array.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


from microbiome._vendor.analysis_toolbox.list import intersection
from microbiome._vendor.analysis_toolbox.function import list_functions
import numpy as np
import dask.array as da
from functools import partial


def return_array_module(arr):

    import numpy as np
    import dask.array as da

    if isinstance(arr, np.ndarray):
        return np
    elif isinstance(arr, da.Array):
        return da


def return_module_func(a, func_name, *args, **kwargs):

    module = return_array_module(a)

    return getattr(module, func_name)(a, *args, **kwargs)


da_funcs = list_functions(da, mod_function=False)


np_funcs = list_functions(np)


overlap_funcs = intersection(da_funcs,
                             np_funcs
                             )


da_funcs.sort()


np_funcs.sort()


overlap_funcs.sort()


for func_name in overlap_funcs:

    vars()[func_name] = partial(return_module_func, func_name=func_name)
