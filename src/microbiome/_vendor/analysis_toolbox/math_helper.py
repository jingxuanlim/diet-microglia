# Vendored from analysis_toolbox (math_helper.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np


def step(step_loc, step_val=None, val_start=-1):
    """Create a step function based on where those steps occur.

    Parameters
    ----------
    step_loc : A binary array indicating where the steps occur. The
               values are in an ascending order and with a step size
               of +1, unless `step_val` is provided.

    step_val : numpy.array or list

               An array of the same size with the values of the steps.
               All the values, besides the ones that occur at the step,
               will be ignored. Values don't have to be strictly
               ascending.

               A list of values that corresponds to the values of the
               steps, before the subsequent step.

    val_start : Starting value of the steps. This only applies when
                step_val is not provided as an array. Can be used as
                pre-first-step value if step_val is a list.


    Returns
    -------
    out :

    """

    result = np.zeros(step_loc.shape)

    if isinstance(step_val, (list, tuple)):
        list_counter = -1
    val = val_start

    for i, t in enumerate(step_loc):

        # at the step locations, change the value either by increasing
        # it, or setting it to the value provided
        if t:

            if step_val is not None:
                if isinstance(step_val, np.ndarray):
                    val = step_val[i]
                elif isinstance(step_val, (list, tuple)):
                    list_counter += 1
                    val = step_val[list_counter]
                else:
                    val += step_val

            else:
                val += 1

        # apply value
        result[i] = val

    return result


def order_of_magnitude(number):
    import math

    if number == 0:
        return number
    else:
        return math.floor(math.log(number, 10))


def percentile_order_of_magnitude_difference(
    arr,
    top_percentile=98,
    bottom_percentile=2,
    debug=False,
):

    from microbiome._vendor.analysis_toolbox.math_helper import order_of_magnitude

    top_value = np.nanpercentile(arr, top_percentile)
    bottom_value = np.nanpercentile(arr, bottom_percentile)
    diff = top_value - bottom_value

    if debug:
        print(
            f"Top value: {top_value}\nBottom value: {bottom_value}\nDifference: {diff}"
        )

    oom_diff = order_of_magnitude(np.abs(diff))

    if debug:
        print(f"Order of magnitude difference: {oom_diff}")

    return oom_diff


def order_of_magnitude_difference(value1, value2, debug=False):

    print(value1, value2)
    diff = value1 - value2
    oom_diff = order_of_magnitude(np.abs(diff))

    if debug:
        print(f"Difference: {diff}")
        print(f"Order of magnitude difference: {oom_diff}")

    return oom_diff


def check_same_sign(arr):
    return np.all(arr > 0) if arr[0] > 0 else np.all(arr < 0)


def find_change(timeseries, first_value, second_value, align="second"):
    """
    Find all instances where the values change from `first_value`
    to the `second_value`.
    The delta will fall on the `second_value` by default.
    """
    first_derivative = (timeseries[:-1] == first_value) * (
        timeseries[1:] == second_value
    )

    if align == "second":
        first_derivative = np.insert(first_derivative, 0, 0)
    elif align == "first":
        first_derivative = np.insert(
            first_derivative, len(first_derivative), 0)

    return first_derivative
