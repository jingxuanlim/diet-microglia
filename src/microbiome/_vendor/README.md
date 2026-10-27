# Vendored code

## `analysis_toolbox`

A subset of [`analysis_toolbox`](https://github.com/jingxuanlim/analysis_toolbox), the author's
general-purpose analysis library, is
included here so that this repository is self-contained.

- **Source:** commit `7f96772069025d136f802c364879703b086dec94`, plus uncommitted
  working-tree changes in `stacks_helper.py` and `visual.py` at the time of vendoring.
- **What is included:** every top-level function, class and constant that is statically
  reachable from the `analysis_toolbox` imports in `notebooks/process-microglia.ipynb` and
  `src/microbiome/microglia.py`, together with the module-level imports and statements they
  need. Classes are included whole. The module layout is unchanged.
- **How it was produced:** definitions were copied verbatim. The only automatic change is that
  `analysis_toolbox.` import paths were rewritten to `microbiome._vendor.analysis_toolbox.`.
  Module-level imports were pruned to the names the retained code uses.

### Manual modifications

Both changes are in `parallel.py::start_cluster`, and each is marked `[vendored edit]` in the
code:

1. The `fish` import is now made only in the Janelia job-queue branch, the only place it is
   used. As a result the `fish` package is not needed anywhere else.
2. On an unrecognised host, the function now starts a plain `dask.distributed.LocalCluster`
   instead of returning `None`. You can configure it with these environment variables:
   - `DASK_MEMORY_LIMIT`: memory per worker. Default `auto`.
   - `DASK_WORKER_SPACE`: spill directory. Default: dask's own default.

   The hostname-specific branches used for the manuscript are unchanged.
