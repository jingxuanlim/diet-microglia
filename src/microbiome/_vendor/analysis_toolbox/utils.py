# Vendored from analysis_toolbox (utils.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import operator
from functools import partial
import numbers
import os
import pandas as pd
import numpy as np
import collections
import dask.array as da
from scipy import stats
from tqdm.auto import tqdm
from IPython.display import display


def combine_objects_if_equal(objects, force=False):
    """
    TODO: is this basically calling np.unique()?
    """

    from microbiome._vendor.analysis_toolbox.utils import (
        check_list_of_arrays_are_equal,
        check_all_elements_equal,
    )

    if isinstance(objects[0], np.ndarray):
        equal = check_list_of_arrays_are_equal(objects)
        print(f"Array, equality check: {equal}")
    else:
        equal = check_all_elements_equal(objects)
        print(f"Not array, equality check: {equal}")

    if equal or force:

        out = objects[0]

    else:
        out = objects

    return out


def rsetattr(obj, attr, val):
    """setattr on nested subobjects / chained properties
    Taken from: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """getattr on nested subobjects / chained properties
    Taken from: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """
    import functools

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr):
    """
    Taken from: https://stackoverflow.com/a/67303315
    """

    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def return_object_with_attributes(objects, ensure_list=False, **kwargs):

    import numpy as np

    obj_bool = []

    for obj in objects:

        obj_bool.append(
            np.all([obj.__dict__.get(key) == value for key, value in kwargs.items()])
        )

    target = np.array(objects)[np.array(obj_bool)]

    if len(target) == 1 and not ensure_list:
        return target[0]
    else:
        return list(target)


def create_necessary_namespaces(obj, attr):

    import types

    subattrs = attr.split(".")
    allattr = ""

    for subattr in subattrs:
        upattr = allattr
        if allattr:
            allattr += "."

        allattr += subattr

        if not rhasattr(obj, allattr):
            print(upattr, subattr)
            setattr(
                rgetattr(obj, upattr) if upattr else obj,
                subattr,
                types.SimpleNamespace(),
            )


def combine_attribute(
    input_objects,
    attr_name,
    combine_mode="combine",
    output_object=None,
    attr_out_name="",
    force=False,
):

    from microbiome._vendor.analysis_toolbox.utils import rgetattr

    all_attrs = [
        rgetattr(obj, attr_name)
        for obj in tqdm(input_objects)
        if rhasattr(obj, attr_name)
    ]

    if len(all_attrs) == 1:
        attr_val = all_attrs[0]
    else:
        if combine_mode == "combine":
            from microbiome._vendor.analysis_toolbox.utils import combine_objects_if_equal

            attr_val = combine_objects_if_equal(all_attrs, force=force)

        elif combine_mode == "concatenate":

            print(f"Concatenating.... {type(all_attrs[0])}")

            if isinstance(all_attrs[0], dict):
                print("Merging dicts")
                from functools import reduce

                attr_val = reduce(merge_dicts, all_attrs)

            else:
                attr_val = all_attrs

    if output_object is not None:

        if not attr_out_name:
            attr_out_name = attr_name

        from microbiome._vendor.analysis_toolbox.utils import rsetattr

        print(f"Setting {attr_out_name} in {type(output_object)}")

        namespaces = ".".join(attr_name.split(".")[:-1])
        if not rhasattr(output_object, namespaces):
            create_necessary_namespaces(output_object, namespaces)

        rsetattr(output_object, attr_out_name, attr_val)

    else:

        return attr_val


def setattrs(_self, **kwargs):
    for k, v in kwargs.items():
        setattr(_self, k, v)


def is_numeric(x):
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)


def apply_to_each_and_every(list_of_objects, function):

    collect = []
    for obj in list_of_objects:
        collect.append(function(obj))

    return function(collect)


def unique(l, return_index=True):
    """
    Taken from: https://www.geeksforgeeks.org/python-ways-to-remove-duplicates-from-list/#
    """
    if return_index:

        # intilize a null list
        unique = []
        indices = []

        # traverse for all elements
        for i, element in enumerate(l):

            # check if exists in unique_list or not
            if element not in unique:
                unique.append(element)
                indices.append(i)

        return unique, indices
    else:
        return [*set(l)]


def concatenate_list_of_lists(list_of_lists):
    import itertools

    return list(itertools.chain.from_iterable(list_of_lists))


def flatten_list_of_lists(t, safe=True):

    if safe:
        if is_list_of_lists(t):
            return [item for sublist in t for item in sublist]
        else:
            return t
    else:
        return [item for sublist in t for item in sublist]


def is_list_of_lists(possible_list_of_lists):

    return isinstance(possible_list_of_lists, list) and isinstance(
        possible_list_of_lists[0], list
    )


def return_chunks_from_list(lst, n):
    """Yield successive n-sized chunks from lst.
    Taken from: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def chunk_list(lst, n):
    return list(return_chunks_from_list(lst, n))


def list_split(lst, num_chunks):
    """
    Split list into num_chunks chunks which are equal-sized
    """

    n = int(np.ceil(len(lst) / num_chunks))

    return chunk_list(lst, n)


def correct_mismatched_starts_and_stops(start_inds, stop_inds):
    """Align start and stop indices"""

    start_inds_type = type(start_inds)
    stop_inds_type = type(stop_inds)

    # work with lists
    if not isinstance(start_inds, list):
        start_inds = list(start_inds)
    else:
        start_inds = copy.deepcopy(start_inds)

    if not isinstance(stop_inds, list):

        stop_inds = list(stop_inds)
    else:
        stop_inds = copy.deepcopy(stop_inds)

    # Start checks
    while len(start_inds) - len(stop_inds) != 0:

        print("Starts: %i Stops: %i" % (len(start_inds), len(stop_inds)))

        for i, (start, stop) in enumerate(zip(start_inds, stop_inds)):

            if start > stop:

                print(i, start, stop)

                if len(stop_inds) > len(start_inds):
                    del stop_inds[i]
                else:
                    del start_inds[i]

                print("Restarting... ")
                break

            if i == min(len(start_inds), len(stop_inds)) - 1:

                if len(stop_inds) > len(start_inds):
                    stop_inds = stop_inds[: len(start_inds) - len(stop_inds)]
                else:
                    start_inds = start_inds[: len(stop_inds) - len(start_inds)]

                print("Done! ")

    if start_inds_type == np.ndarray:
        start_inds = np.array(start_inds)
    elif not start_inds_type == list:
        raise Exception("Don't know how to transform 'start_inds'")

    if stop_inds_type == np.ndarray:
        stop_inds = np.array(stop_inds)
    elif not stop_inds_type == list:
        raise Exception("Don't know how to transform 'stop_inds'")

    return start_inds, stop_inds


def now_str():

    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def tryfloat(x):
    """
    Converts an object to a float and returns it if possible; returns original
    object if not
    """
    try:
        return float(x)
    except:
        return x


def zip_and_join(*lists_of_str, sep=None):

    combined_list_of_tuples = list(zip(*lists_of_str))

    if sep is not None:
        return [sep.join(clist) for clist in combined_list_of_tuples]
    else:
        return combined_list_of_tuples


def check_binary(a, mode="simple"):

    if mode == "simple":
        return ((a == 0) | (a == 1)).all()
    if mode == "alt":
        return ~((a != 0) & (a != 1)).any()
    if mode == "count_nonzero":
        np.count_nonzero((a != 0) & (a != 1)) == 0
    if mode == "count_zero":
        a.size == np.count_nonzero((a == 0) | (a == 1))


def search_list_of_str(list_of_str, regex):

    import re

    r = re.compile(regex)
    matches = list(filter(r.match, list_of_str))

    return matches


def sample_indices(ind_arr, ind_fs, target_fs, align="left"):
    """
    ind_arr [array]: numpy array containing the indices; strictly ascending or descending
    ind_fs [float]: sampling rate of the indices
    target_fs [float]: final sampling rate; must be smaller than ind_fs
    align [str/float]: if float, align to a certain value
                       if 'left' or 'right', align to the first or last value
    """

    resampling_interval = int(np.ceil(ind_fs / target_fs))

    ind_arr = np.sort(ind_arr)

    if isinstance(align, str):

        if align == "left":
            target_arr = ind_arr[::resampling_interval]
        elif align == "right":
            target_arr = ind_arr[::-1][::resampling_interval][::-1]

    else:

        neg = ind_arr[ind_arr < align]
        pos = ind_arr[ind_arr >= align]

        tar_neg = -neg[::-1][resampling_interval - 1 :: resampling_interval]
        tar_pos = pos[::resampling_interval]

        target_arr = np.concatenate((-tar_neg[::-1], tar_pos))

    return target_arr


def change_file_ext(filepath, new_ext):

    import pathlib

    return pathlib.Path(filepath).with_suffix(new_ext).as_posix()


def generate_random_filepath(root_dir="", verbose=True):

    import uuid

    filename = str(uuid.uuid4())

    if not root_dir:  # if no root_dir is provided
        import socket

        hostname = socket.gethostname()
        if verbose:
            print(f"Generating filepath on {hostname}...")

        # if on the janelia cluster, start a non-local cluster
        if hostname == "e06u03.int.janelia.org":
            root_dir = "/scratch/limj2/temp"
        elif hostname == "ahrenslab-ws1" or hostname == "ahrensm-ws2":
            root_dir = "/scratch/limj2/temp"
        elif hostname == "picasso":
            root_dir = "/data1/limj2/temp"
        elif hostname[7:] == "int.janelia.org":
            root_dir = "/nrs/ahrens/jing/temp"
        else:
            root_dir = "/nrs/ahrens/jing/temp"

    return root_dir + "/" + filename


def ensure_extension(path, ext):

    filename, fileext = os.path.splitext(path)

    if fileext:
        if fileext != ext:
            path = filename + "." + ext
    else:
        path = path + "." + ext

    return path


def common_root(dir_list):
    """Find common root directory"""
    import os

    return os.path.dirname(os.path.commonprefix(dir_list))


def ensure_end_in_slash(dir_path):

    if dir_path[-1] not in ["/", "\\"]:

        return path_slash(dir_path)

    return dir_path


def ensure_not_end_in_slash(dir_path):

    if dir_path[-1] in ["/", "\\"]:

        return dir_path[:-1]

    return dir_path


def path_slash(path):
    """
    Finds out the direction of slash and appends it to the end of the
    directory path.

    Parameters
    ----------
    path [str]: path of directory without a trialing slash


    Returns
    -------
    out :

    """

    import re

    regex = re.compile("[@_!#$%^&*()<>?/\|}{~:]")
    out = regex.search(path)
    slash = out.group()

    if path[-1] != slash:
        return path + slash
    else:
        return path


def change_path_root(original_path, immediate_dir, root_dir):
    """
    Takes the directory of a pathlib path and prepends with another root
    directory.

    Parameters
    ----------
    original_path [str/pathlib]: original_path to be "fixed"
    e.g. '/x/y/z'

    immediate_dir [str]: name of directory included in both the original_path
                         and the root_dir
    e.g. 'y'

    root_dir [str]:
    e.g. '/a/y'

    Returns
    -------
    out :
    e.g. '/a/y/z'

    """
    import pathlib

    if type(original_path) != pathlib.PosixPath:
        full_path = pathlib.Path(original_path)
    else:
        full_path = original_path

    index = full_path.parts.index(immediate_dir)

    if type(original_path) != pathlib.PosixPath:
        return pathlib.Path(root_dir).joinpath(*full_path.parts[index + 1 :]).as_posix()
    else:
        return pathlib.Path(root_dir).joinpath(*full_path.parts[index + 1 :])


def change_root_dir_in_path(path, replace_with):
    list_of_strings = path.split("/")
    filename = list_of_strings[-1]

    # remove trailing slash if any
    if replace_with[-1] == "/":
        replace_with = replace_with[:-1]

    return "/".join([replace_with, filename])


def checkEqual3(lst):
    return lst[1:] == lst[:-1]


def countdown(t, pre_msg="Counting down...", post_msg="Time is up!"):

    import time

    while t:
        mins, secs = divmod(t, 60)
        timer = "{:02d}:{:02d}".format(mins, secs)
        if t == 1:
            end = "\n"
        else:
            end = "\r"
        print(pre_msg, timer, end=end)
        time.sleep(1)
        t -= 1

    print(post_msg)


def find_duplicates_in_list(input_list):
    import collections

    return [
        item for item, count in collections.Counter(input_list).items() if count > 1
    ]


def find_duplicates_in_arr(arr):
    u, c = np.unique(arr, return_counts=True)
    dup = u[c > 1]
    return dup


def find_duplicates(arr_like):
    if isinstance(arr_like, np.ndarray):
        return find_duplicates_in_arr(arr_like)
    elif isinstance(arr_like, list):
        return find_duplicates_in_list(arr_like)


def cust_range(*args, rtol=1e-05, atol=1e-08, include=[True, False]):
    """
    Combines numpy.arange and numpy.isclose to mimic
    open, half-open and closed intervals.
    Avoids also floating point rounding errors as with
    >>> numpy.arange(1, 1.3, 0.1)
    array([1. , 1.1, 1.2, 1.3])

    args: [start, ]stop, [step, ]
        as in numpy.arange
    rtol, atol: floats
        floating point tolerance as in numpy.isclose
    include: boolean list-like, length 2
        if start and end point are included

    Taken from: https://stackoverflow.com/a/57321916
    """
    # process arguments
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    else:
        assert len(args) == 3
        start, stop, step = tuple(args)

    # determine number of segments
    n = (stop - start) / step + 1

    # do rounding for n
    if np.isclose(n, np.round(n), rtol=rtol, atol=atol):
        n = np.round(n)

    # correct for start/end is exluded
    if not include[0]:
        n -= 1
        start += step
    if not include[1]:
        n -= 1
        stop -= step

    return np.linspace(start, stop, int(n))


def crange(*args, **kwargs):
    """Like np.arange, but includes the stop point
    Taken from: https://stackoverflow.com/a/57321916
    """
    return cust_range(*args, **kwargs, include=[True, True])


def group_list_of_strings(list_of_strings, number_of_substring_blocks=1, debug=False):

    from itertools import groupby

    # sort list
    # essential for grouping
    list_of_strings.sort()

    if debug:
        # printing the original list
        print("The original list is : " + str(list_of_strings))

    # using lambda + itertools.groupby() + split()
    # group similar substrings
    res = [
        list(i)
        for j, i in groupby(
            list_of_strings,
            lambda a: "_".join(a.split("_")[:number_of_substring_blocks]),
        )
    ]

    if debug:
        # printing result
        print("The grouped list is : " + str(res))

    return res


def attributes_contain(obj, contains=[], filter_relationship=np.logical_or.reduce):

    filter_list = []

    for contain in contains:
        filter_list.append([contain in key for key in obj.__dict__.keys()])

    criteria = filter_relationship(filter_list)

    return list(np.array(list(obj.__dict__.keys()))[criteria])


def return_object_containing_attr_of_value(object_list, attr_name, attr_value):

    object_bool = []

    for obj in tqdm(object_list):
        if hasattr(obj, attr_name):
            object_bool.append(getattr(obj, attr_name) == attr_value)

        else:
            object_bool.append(False)

    objects = np.array(object_list)[np.array(object_bool)]

    if len(objects) == 1:
        return objects[0]
    else:
        return objects


def merge_dicts(dict1, dict2):
    """
    Merge two dictionaries.
    """
    return {**dict1, **dict2}


def count_number_of_nested_lists(l):
    return sum(1 + count_number_of_nested_lists(i) for i in l if isinstance(i, list))


def check_all_elements_equal(arr):
    return np.all(arr == arr[0])


nanmode = partial(stats.mode, nan_policy="omit")


nansem = partial(stats.sem, nan_policy="omit")


nanfuncs = {
    "nanprod": np.nanprod,
    "nansum": np.nansum,
    "nancumprod": np.nancumprod,
    "nancumsum": np.nancumsum,
    "nanmax": np.nanmax,
    "nanmin": np.nanmin,
    "nanpercentile": np.nanpercentile,
    "nanquantile": np.nanquantile,
    "nanmode": nanmode,
    "nansem": nansem,
    "nanmean": np.nanmean,
    "nanmedian": np.nanmedian,
}


def boolean_indexing(v, fillval=np.nan, pad="back", dtype=None):
    """

    For a list of arrays with difference in length in the first axis,
    pad the shorter arrays `fillval` and return the stacked array.

    Example
    -------
    Something like the following will end up with a shape (8, 67, 21, 512, 512)

    [item.shape for item in v] = [(67, 21, 512, 512),
                                  (67, 21, 512, 512),
                                  (67, 21, 512, 512),
                                  (67, 21, 512, 512),
                                  (16, 21, 512, 512),
                                  (28, 21, 512, 512),
                                  (5, 21, 512, 512),
                                  (57, 21, 512, 512)]

    NOTE: not sure why, but only works with fillval = np.nan
    I think indexing like this (i.e. with booleans of the same shape)
    works with reading and not writing. Writing with nans seems to be
    fine.
    """
    lens = np.array([len(item) for item in v])
    max_len = lens.max()
    lenv = len(lens)

    mask = lens[:, None] > np.arange(max_len)

    if pad == "front":
        mask = mask[..., ::-1]

    if dtype is None:
        if v[0].dtype == object:
            dtype = v[0].dtype

    out = np.full((lenv, max_len, *v[0].shape[1:]), fill_value=fillval, dtype=dtype)
    out[mask] = np.concatenate(v)

    return out


def check_array_shapes_identical(list_of_arr):
    """Check that all arrays in list have the same shape"""
    return np.all([x.shape[0] == list_of_arr[0].shape[0] for x in list_of_arr])


def check_list_of_arrays_are_equal(list_of_arr):
    return np.all([np.array_equal(arr, list_of_arr[0]) for arr in list_of_arr])


def isnan(var):

    if isinstance(var, float):
        import math

        return math.isnan(var)

    elif isinstance(var, np.ndarray) and var.shape == ():
        return np.isnan(var)

    else:
        return False


def num_to_nan(a, val=0):

    a[a == val] = np.nan

    return a


def float_gcd(a, b, rtol=1e-05, atol=1e-08):
    """
    Taken from: https://stackoverflow.com/a/45325587
    """
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a


def solve_ratio(ratio):

    if isinstance(ratio, str):
        numbers = [int(i) for i in ratio.split(":")]
    else:
        numbers = ratio

    from functools import reduce
    from math import gcd

    denominater = reduce(float_gcd, numbers)
    solved = [int(i / denominater) for i in numbers]

    if isinstance(ratio, str):
        return ":".join(str(i) for i in solved)
    else:
        return solved


def _replace_nan(a, val):
    """
    COPIED wholesale from a function in numpy's nanfunction.py of
    the same name.

    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.

    Note that scalars will end up as array scalars, which is important
    for using the result as the value of the out argument in some
    operations.

    Parameters
    ----------
    a : array-like
        Input array.
    val : float
        NaN values are set to val before doing the operation.

    Returns
    -------
    y : ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return None.

    """
    a = np.asanyarray(a)

    if a.dtype == np.object_:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        mask = np.not_equal(a, a, dtype=bool)
    elif issubclass(a.dtype.type, np.inexact):
        mask = np.isnan(a)
    else:
        mask = None

    if mask is not None:
        a = np.array(a, subok=True, copy=True)
        np.copyto(a, val, where=mask)

    return a, mask


def center_minmax(arr, min_pctl=None, max_pctl=None):

    if min_pctl is not None and max_pctl is not None:
        amin = np.nanpercentile(arr, min_pctl)
        amax = np.nanpercentile(arr, max_pctl)
    else:
        # limits are given
        assert len(arr) == 2
        amin = arr[0]
        amax = arr[1]

    max_a = np.nanmax((np.abs(amin), np.abs(amax)))

    vmin = -max_a
    vmax = max_a

    return vmin, vmax


def normalise(a, a_min=None, a_max=None):
    """Normalize to [0, 1]"""

    from microbiome._vendor.analysis_toolbox.array import nanmin, nanmax

    if a_min is None:
        a_min = nanmin(a)
    if a_max is None:
        a_max = nanmax(a)

    return (a - a_min) / (a_max - a_min)


def gain_arange(start, stop, gain, step=1):

    return np.arange(int(start * gain), int(stop * gain), step)


def dask_array_split(ary, indices_or_sections, axis=0, return_div_points=False):
    """
    ADAPTED FROM: https://github.com/numpy/numpy/blob/4adc87dff15a247e417d50f10cc4def8e1c17a03/numpy/lib/shape_base.py#L740
    Split an array into multiple sub-arrays.
    Please refer to the ``split`` documentation.  The only difference
    between these functions is that ``array_split`` allows
    `indices_or_sections` to be an integer that does *not* equally
    divide the axis. For an array of length l that should be split
    into n sections, it returns l % n sub-arrays of size l//n + 1
    and the rest of size l//n.
    See Also
    --------
    split : Split array into multiple sub-arrays of equal size.
    Examples
    --------
    >>> x = np.arange(8.0)
    >>> np.array_split(x, 3)
    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.])]
    >>> x = np.arange(9)
    >>> np.array_split(x, 4)
    [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]
    """

    import numpy.core.numeric as _nx

    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError("number sections must be larger than 0.") from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = (
            [0] + extras * [Neach_section + 1] + (Nsections - extras) * [Neach_section]
        )
        div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

    sub_arys = []
    sary = da.swapaxes(ary, axis, 0)  # swap axes to easily index
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        # index the first axis
        sub_arys.append(da.swapaxes(sary[st:end], axis, 0))

    if return_div_points:
        return sub_arys, div_points
    else:
        return sub_arys


def get_chunksize_from_chunks(arr, chunks, mode="fast"):
    """Get slices.

    Get slices from chunks by a dask array intermediary.
    """
    import dask.array as da

    if mode == "fast":
        target_nbytes = parse_size("100 MB")
        npixels_limit = target_nbytes / arr.itemsize

        import numbers

        find_numbers = list(map(lambda x: isinstance(x, numbers.Number), chunks))
        set_npixels = np.multiply.reduce(
            np.array(chunks)[np.array(find_numbers)].astype("int")
        )

        existing_ndim = np.sum(find_numbers)
        remaining_ndims = arr.ndim - existing_ndim

        chunksize = int(
            np.floor((npixels_limit / set_npixels) ** (1 / remaining_ndims))
        )

        new_chunks = np.array(chunks)
        new_chunks[np.logical_not(find_numbers)] = chunksize
        new_chunks = list(new_chunks.astype("int"))
        # convert to python int type because np.int64 doesn't work
        new_chunks = tuple([int(chunksize) for chunksize in new_chunks])

        return new_chunks

    elif mode == "full":
        # might take a while
        # 1min 21s for arr of size (53, 185, 53, 139, 230)
        dask_arr = da.from_array(arr, chunks=chunks)

        return dask_arr.chunksize

    else:

        raise Exception("mode not recognized!")


def ensure_filepath(filepath):

    if not filepath:
        filepath = generate_random_filepath()
    else:
        if os.path.isdir(filepath):
            filepath = generate_random_filepath(root_dir=filepath)

    return filepath


def zscore(series, mean=None, std=None):

    series.replace([np.inf, -np.inf], np.nan, inplace=True)

    if mean is None:
        mean = series.mean()

    if std is None:
        std = series.std(ddof=0)

    return (series - mean) / std


def operate_on_first_dim(arr, var, operation=operator.eq):

    if arr.ndim == 2:

        return np.any(arr == var, axis=1)

    elif arr.ndim == 1:

        return arr == var


def pad(series, column, pad_loc, pad_len):
    """
    column [str]: name of column to be padded
    pad_loc [str]: 'front', 'back', 'both'
    pad_len [int/list]
    """

    interpolated = series[column]

    from microbiome._vendor.analysis_toolbox.utils import isnan

    if isnan(interpolated):
        return np.nan
    else:
        if pad_loc == "back":
            padded = np.pad(
                interpolated,
                mode="constant",
                pad_width=((0, 0), (0, pad_len)),
                constant_values=np.nan,
            )
        elif pad_loc == "front":
            padded = np.pad(
                interpolated,
                mode="constant",
                pad_width=((0, 0), (pad_len, 0)),
                constant_values=np.nan,
            )
        elif pad_loc == "both":
            padded = np.pad(
                interpolated,
                mode="constant",
                pad_width=((0, 0), (pad_len[0], pad_len[1])),
                constant_values=np.nan,
            )

        return padded


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}


def parse_size(size):
    """
    Taken from: https://stackoverflow.com/a/60708339
    based on https://stackoverflow.com/a/42865957/2002471
    """
    import re

    size = size.upper()
    # print("parsing size ", size)
    if not re.match(r" ", size):
        size = re.sub(r"([KMGT]?B)", r" \1", size)
    number, unit = [string.strip() for string in size.split()]
    return int(float(number) * units[unit])


def profile_workspace(return_values="Gb"):
    """
    Calculate in Gb
    """

    import psutil

    values = psutil.virtual_memory()

    print("{:>30}: {:>2} Gb".format("Total", values.total >> 30))
    print("{:>30}: {:>2} Gb".format("Available", values.available >> 30))
    print("{:>30}: {:>2} %".format("Percent", values.percent))
    print("{:>30}: {:>2} Gb".format("Used", values.used >> 30))
    print("{:>30}: {:>2} Gb".format("Free", values.free >> 30))
    print("{:>30}: {:>2} Gb".format("Active", values.active >> 30))
    print("{:>30}: {:>2} Gb".format("Inactive", values.inactive >> 30))
    print("\n")

    if return_values:
        if return_values == "Gb":
            Gb_dict = {}

            for attr in ["total", "available", "used", "free", "active", "inactive"]:
                Gb_dict[attr] = getattr(values, attr) >> 30

            from types import SimpleNamespace

            return SimpleNamespace(**Gb_dict)

        else:
            return values


def extended_glob(search_term, directory):
    """ """
    print("Using regular expression instead of glob")

    import os
    import re

    files = []
    for fname in tqdm(os.listdir(directory)):
        if re.match(search_term, fname):
            files.append(fname)

    return files


def find_files(
    directory,
    ext="*",
    grep="",
    grep_complete=False,
    verbose=False,
    verbose_slicer=None,
    eglob=False,
    compute_paths=False,
    compute_basenames=False,
    compute_filesizes=False,
    compute_creationdates=False,
    return_type="dict",
):

    if ext is None:
        if verbose:
            print(f"Looking for directories...")
        ext = ""
    elif ext[0] != ".":
        ext = "." + ext

    from glob import glob
    import os

    if directory[-1] != "/":
        directory = directory + "/"

    # "one or more" is not supported by glob
    # https://stackoverflow.com/a/37864354
    if eglob or "+" in grep:
        search_term = grep + ext
        files = extended_glob(search_term, directory)
    else:
        if not grep_complete:
            file = "*" + grep + "*"
        else:
            file = grep
        search_term = directory + file + ext
        files = glob(search_term)

    files.sort()

    if verbose:
        print("Searching for %s ...\n" % search_term)
        print("%i files found!" % len(files), end="\n\n")

    file_info = {}
    if compute_paths:
        file_info["path"] = files

    from tqdm.notebook import tqdm

    # extract file names
    if compute_basenames:
        if verbose:
            print("Computing base names...")
        filenames = list(map(path_leaf, tqdm(files)))
        file_info["basename"] = filenames
    if compute_filesizes:
        if verbose:
            print("Computing file sizes...")
        filesizes = list(map(file_size, tqdm(files)))
        file_info["filesize"] = filesizes
    if compute_creationdates:
        if verbose:
            print("Computing creation dates...")
        filecreationdates = list(map(creation_datetime, tqdm(files)))
        file_info["creationdate"] = filecreationdates

    import pandas as pd

    file_info_df = pd.DataFrame.from_dict(file_info)

    if verbose:

        from IPython.display import Image, display

        if verbose_slicer is None:
            verbose_slicer = slice(len(files))

        display(file_info_df.iloc[verbose_slicer])

    if return_type == "dict":
        return file_info
    elif return_type == "df":
        return file_info_df


def return_dirs_with_contents(dirs, **kwargs):

    exists_bool = []

    for dir_ in dirs:
        exists_bool.append(file_exists_in_dir(dir_, **kwargs))

    return list(np.array(dirs)[np.array(exists_bool)])


def file_exists_in_dir(dir_, **kwargs):

    from microbiome._vendor.analysis_toolbox.utils import find_files

    if not isinstance(dir_, str):
        dir_ = dir_.as_posix()

    found_files = find_files(dir_, compute_paths=True, **kwargs)["path"]

    return len(found_files) > 0


def basename(path_):
    """
    Keep applying os.path.split until basename is found
    """

    head, tail = os.path.split(path_)

    while not tail:
        head, tail = os.path.split(head)

    return tail


def path_leaf(path):
    """
    https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    """
    import ntpath

    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def file_size(path_to_file):

    inode = os.stat(path_to_file)
    return sizeof_fmt(inode.st_size)


def creation_time(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    import os
    import platform

    if platform.system() == "Windows":
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


def creation_datetime(path_to_file):

    return parse_timestamp(creation_time(path_to_file))


def parse_timestamp(timestamp):

    import datetime

    return datetime.datetime.fromtimestamp(timestamp)


def symlink(target, link_name, overwrite=False):
    """
    Create a symbolic link named link_name pointing to target.
    If link_name exists then FileExistsError is raised, unless overwrite=True.
    When trying to overwrite a directory, IsADirectoryError is raised.
    """

    import os
    import tempfile

    if not overwrite:
        os.symlink(target, linkname)
        return

    # os.replace() may fail if files are on different filesystems
    link_dir = os.path.dirname(link_name)

    # Create link to target with temporary filename
    while True:
        temp_link_name = tempfile.mktemp(dir=link_dir)

        # os.* functions mimic as closely as possible system functions
        # The POSIX symlink() returns EEXIST if link_name already exists
        # https://pubs.opengroup.org/onlinepubs/9699919799/functions/symlink.html
        try:
            os.symlink(target, temp_link_name)
            break
        except FileExistsError:
            pass

    # Replace link_name with temp_link_name
    try:
        # Pre-empt os.replace on a directory with a nicer message
        if os.path.isdir(link_name):
            raise IsADirectoryError(
                f"Cannot symlink over existing directory: '{link_name}'"
            )
        os.replace(temp_link_name, link_name)
    except:
        if os.path.islink(temp_link_name):
            os.remove(temp_link_name)
        raise


def any_substring_in_string(string, substrings):

    return any(substring in string for substring in substrings)


def return_element_with_substring(list_, substring):

    result = [l for l in list_ if substring in l]

    if len(result) == 1:
        return result[0]
    else:
        return result


def zscore_and_multiply(arrs):

    from scipy.stats import zscore

    return np.multiply.reduce(
        [zscore(arr, axis=None, nan_policy="omit") for arr in arrs]
    )


def safe_argfunc(arr, func, axis=0, keepdims=False):

    # detect all nan slices
    if np.sum(np.all(np.isnan(arr), axis=axis, keepdims=keepdims)):
        from microbiome._vendor.analysis_toolbox import utils

        module = utils
    else:
        module = np

    return getattr(module, func)(arr, axis=axis, keepdims=keepdims)


def nan_take_along_axis(arr, indices, axis):

    indices, mask = _replace_nan(indices, 0)
    indices = indices.astype("int")

    indexed = np.take_along_axis(arr, indices, axis)

    if mask is not None:
        indexed[mask] = np.nan

    return indexed


def index_based_on_projection(arr_to_proj, axis=0, arrs_to_index=[], proj_type="min"):

    proj = safe_argfunc(arr_to_proj, f"nanarg{proj_type}", axis=axis, keepdims=True)

    if isinstance(arrs_to_index, list):
        indexed = []
        for arr_to_index in arrs_to_index:
            indexed.append(
                np.squeeze(nan_take_along_axis(arr_to_index, proj, axis=axis))
            )

    else:
        indexed = np.squeeze(nan_take_along_axis(arrs_to_index, proj, axis=axis))

    return indexed
