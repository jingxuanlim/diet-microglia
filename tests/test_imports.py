import importlib
import pkgutil

import numpy as np
import pytest

import microbiome._vendor.analysis_toolbox as vendored_toolbox
from microbiome.microglia import _is_expected_timestamp_array, correct_timestamps

VENDORED_MODULES = [
    m.name
    for m in pkgutil.walk_packages(vendored_toolbox.__path__, vendored_toolbox.__name__ + ".")
]

NOTEBOOK_IMPORTS = [
    ("microbiome.microglia", "correct_timestamps"),
    ("microbiome.microglia", "extract_microglia_candidates_using_complex_outline_computation"),
    ("microbiome.microglia", "extract_microglia_candidates_using_simple_outline_computation"),
    ("microbiome.microglia", "extract_and_characterize_microglia"),
    ("microbiome.microglia", "TEMP_DIR"),
    ("microbiome._vendor.analysis_toolbox.utils", "now_str"),
    ("microbiome._vendor.analysis_toolbox.fileio", "load_gsheet"),
    ("microbiome._vendor.analysis_toolbox.fileio", "make_local_copy"),
    ("microbiome._vendor.analysis_toolbox.fileio", "chunk_and_saveas_zarr"),
    ("microbiome._vendor.analysis_toolbox.dataorg.dataset_collection", "initialize_dataset_collection"),
    ("microbiome._vendor.analysis_toolbox.parallel", "dask_client_compute"),
    ("microbiome._vendor.analysis_toolbox.stacks_helper", "plot_planes"),
    ("microbiome._vendor.analysis_toolbox.dataset.functional_imaging", "load_func_im"),
    ("bigstream.motion_correct", "motion_correct"),
    ("bigstream.motion_correct", "resample_frames"),
]


@pytest.mark.parametrize("module", VENDORED_MODULES)
def test_vendored_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module, name", NOTEBOOK_IMPORTS)
def test_notebook_imports_resolve(module, name):
    assert hasattr(importlib.import_module(module), name)


def test_bigstream_has_patched_motion_correct():
    import inspect

    from bigstream.motion_correct import motion_correct

    assert "indices" in inspect.signature(motion_correct).parameters


def two_epoch_series(short_h=0.063, short_total_h=1.0, long_h=0.5, long_total_h=4.0):
    short = np.arange(0, short_total_h, short_h)
    long = short[-1] + long_h + np.arange(0, long_total_h, long_h)
    return np.concatenate([short, long])


def test_correct_timestamps_returns_valid_series_unchanged():
    series = two_epoch_series()
    assert _is_expected_timestamp_array(series)
    np.testing.assert_array_equal(correct_timestamps(series, metadata_or_periods=None), series)


def test_correct_timestamps_rebuilds_all_zero_series_from_manual_durations():
    corrected = correct_timestamps(
        np.zeros(24),
        metadata_or_periods=None,
        manual_intervals_h=[0.063, 0.5],
        manual_durations_h=[1.0, 96.0],
    )
    assert corrected.shape == (24,)
    assert _is_expected_timestamp_array(corrected)
