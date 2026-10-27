# Vendored from analysis_toolbox (parallel.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import multiprocessing.pool as mpp
import os
import numpy as np
import dask.array as da
import xarray as xr
import dask.bag as db
from tqdm import tqdm_notebook as tqdm
from typing import Union, Optional
from collections.abc import Callable


def num_cpu():

    import multiprocessing

    return multiprocessing.cpu_count()


def return_cores_info(verbose=False):

    import os
    import multiprocessing

    slots_requested = int(os.environ["LSB_DJOB_NUMPROC"])
    virtual_cores = os.cpu_count()

    if verbose:
        print(
            f'int(os.environ["LSB_DJOB_NUMPROC"]): {int(os.environ["LSB_DJOB_NUMPROC"]) }'
        )
        print(f"len(os.sched_getaffinity(0)): {len(os.sched_getaffinity(0))}")
        print(f"os.cpu_count(): {os.cpu_count()}")
        print(f"multiprocessing.cpu_count(): {multiprocessing.cpu_count()}")

    return slots_requested, virtual_cores


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


def start_cluster(
    cores=1,
    processes=1,
    walltime="12:00",
    force_local=False,
    store_logs=False,
    spillover=False,
    verbose=False,
    worker_lifetime=None,
    worker_lifetime_stagger=None,
    worker_lifetime_restart=False,
    **kwargs,
):

    # Distributed computing
    from dask.distributed import LocalCluster
    import socket

    hostname = socket.gethostname()
    if verbose:
        print(f"Kernel currently running on {hostname}")

    # if on the janelia cluster, start a non-local cluster
    if hostname == "e06u03.int.janelia.org":

        dask_tmp = "/scratch/limj2/dask-worker-space"
        memory_limit = str(1007 / np.min((processes, 256))) + "GB"  # unlimited

        cluster = LocalCluster(
            processes=True,
            threads_per_worker=cores,
            memory_limit=memory_limit,
            local_directory=dask_tmp,
            n_workers=processes,
            **kwargs,
        )

    # if on the janelia cluster, start a non-local cluster
    elif hostname == "ahrenslab-ws1.hhmi.org":

        dask_tmp = "/home/limj2@hhmi.org/dask-worker-space"
        memory_limit = str(1007 / np.min((processes, 96))) + "GB"  # unlimited

        cluster = LocalCluster(
            processes=True,
            threads_per_worker=cores,
            memory_limit=memory_limit,
            local_directory=dask_tmp,
            n_workers=processes,
            **kwargs,
        )

    # if on the janelia cluster, start a non-local cluster
    elif hostname == "ahrensm-ws2.hhmi.org":

        dask_tmp = "/scratch/limj2/dask-worker-space"
        memory_limit = str(754 / np.min((processes, 72))) + "GB"  # unlimited

        cluster = LocalCluster(
            processes=True,
            threads_per_worker=cores,
            memory_limit=memory_limit,
            local_directory=dask_tmp,
            n_workers=processes,
            **kwargs,
        )

    elif hostname[7:] == "int.janelia.org":

        if spillover:
            dask_tmp = "/nrs/ahrens/jing/cellSeg/dask-worker-space"
        else:
            dask_tmp = None

        if force_local:

            slots_requested, _ = return_cores_info()
            total_memory = 15 * slots_requested
            memory_limit = str(total_memory // processes) + "GB"

            cluster = LocalCluster(
                processes=True,
                threads_per_worker=cores,
                n_workers=processes,
                memory_limit=memory_limit,
                local_directory=dask_tmp,
                **kwargs,
            )

        else:

            #   15 GB is the limit specified by cluster admins
            memory_limit = cores * 15

            # set directory for storing logs
            if store_logs:
                job_extra = ["-o /nrs/ahrens/jing/bsub_logs/%J.log"]
            else:
                job_extra = ["-o /dev/null"]

            # set lifetime extras
            if worker_lifetime is not None or worker_lifetime_stagger is not None:
                extra = []

                if worker_lifetime is not None:
                    extra.append("--lifetime")
                    extra.append(worker_lifetime)
                if worker_lifetime_stagger is not None:
                    extra.append("--lifetime-stagger")
                    extra.append(worker_lifetime_stagger)

                if worker_lifetime_restart:
                    extra.append("--lifetime-restart")

            else:
                extra = None

            # [vendored edit] imported here so `fish` is only needed on the Janelia cluster
            from fish.util.distributed import get_jobqueue_cluster

            cluster = get_jobqueue_cluster(
                cores=cores,  # total number of cores per job
                memory=str(memory_limit) + "GB",
                processes=processes,  # total number of processes per job
                walltime=walltime,
                extra=extra,
                job_extra=job_extra,
                local_directory=dask_tmp,
                **kwargs,
            )

    # if on a workstation, start a local cluster and request workers=cores
    elif hostname == "ahrensm-ws2":

        dask_tmp = "/opt/data/limj2/dask-worker-space"
        memory_limit = str(754 / np.min((processes, 72))) + "GB"  # unlimited

        cluster = LocalCluster(
            processes=True,
            memory_limit=memory_limit,
            local_dir=dask_tmp,
            n_workers=processes,
            **kwargs,
        )

    elif hostname == "ahrenslab-ws1":

        dask_tmp = (
            "/scratch/limj2/dask-worker-space"  # '/opt/data/limj2/dask-worker-space'
        )
        memory_limit = (
            str(1006 / np.min((processes, 96))) + "GB"
        )  # 1006//96  ## unlimited

        cluster = LocalCluster(
            processes=True,
            memory_limit=memory_limit,
            local_dir=dask_tmp,
            n_workers=processes,
            **kwargs,
        )

    elif hostname == "picasso":

        dask_tmp = "/opt/data/limj2/dask-worker-space"
        memory_limit = str(252 / np.min((processes, 32))) + "GB"

        cluster = LocalCluster(
            processes=True,
            memory_limit=memory_limit,
            local_directory=dask_tmp,
            n_workers=processes,
            **kwargs,
        )

    elif hostname == "Pangu":

        dask_tmp = "/mnt/data/dask-worker-space"
        from microbiome._vendor.analysis_toolbox.utils import profile_workspace
        RAM = profile_workspace(return_values="Gb")

        print(f"RAM available: {RAM.available}")
        memory_limit = str((RAM.available / np.min((processes, 24)))*processes) + "GB"  # unlimited
        memory_limit = "40GB"
        print(f"Memory limit: {memory_limit}")
        cluster = LocalCluster(
            processes=True,
            threads_per_worker=cores,
            memory_limit=memory_limit,
            local_directory=dask_tmp,
            n_workers=processes,
            **kwargs,
        )

    else:
        # [vendored edit] fall back to a plain local cluster instead of returning None
        print("Compute cluster not identified; starting a generic LocalCluster")

        cluster = LocalCluster(
            processes=True,
            threads_per_worker=cores,
            memory_limit=os.environ.get("DASK_MEMORY_LIMIT", "auto"),
            local_directory=os.environ.get("DASK_WORKER_SPACE"),
            n_workers=processes,
            **kwargs,
        )

    if verbose:
        print_client_links(cluster)
        # print(cluster.job_script())

    return cluster


def print_client_links(cluster):
    print(f"Scheduler: {cluster.scheduler_address}")
    print(f"Dashboard link: {cluster.dashboard_link}")


def calculate_num_processes(max_task_size, num_cores_per_worker):
    """Calculate optimal number of processes based on RAM limitations
    Number of processes capped at number of cores, assuming that one core should
    only work on one process.
    Only makes sense for LocalCluster
    """
    import os
    from microbiome._vendor.analysis_toolbox.utils import profile_workspace

    RAM = profile_workspace(return_values="Gb")

    max_cores = int(np.clip(RAM.available / max_task_size, None, os.cpu_count()))
    num_processes = max_cores // num_cores_per_worker

    return num_processes


def dask_client_compute(
    mappable: Union[list, np.ndarray, da.Array],
    mappable_id: str = "data",
    # ------------------------------
    # (1) Results-related arguments
    # ------------------------------
    zarr_path: str = "",
    # -------------------------------
    # (2) Function-related arguments
    # -------------------------------
    func: Union[Callable, str] = None,
    func_args: tuple = (),
    func_kwargs: dict = {},
    # required when mappable is an array and mappable_id = "data"
    apply_along_axis_kwargs: dict = {"axis": 1, "dtype": None, "shape": None},
    # -----------------------------
    # (3) Client-related arguments
    # -----------------------------
    force_local: bool = True,
    max_task_size: Optional[int] = None,
    num_cores_per_worker: int = 1,
    num_workers: Optional[int] = None,
    return_done: bool = False,
    compute_kwargs: dict = {},
    # --------------------------------
    # (4) Architecture of computation
    # --------------------------------
    loop_nchunk: Optional[int] = None,
    reduce_task_graph: bool = False,
):
    """
    mappable: something to compute, typically
              a list of strings or da.Array where a function would work on,
              or an already defined da.Arrray
    mappable_id: ['data', 'index']

    (1) Results-related arguments
    ------------------------------
    zarr_path: persistence of the result

    (2) Function-related arguments
    -------------------------------
    func: function to be applied with da.apply_along_axis when applied to a da.Array
          or client.map when applied to a list of strings
    func_args: function arguments
    func_kwargs: function keyword arguments
    apply_along_axis_kwargs: dictionary containing kwargs for da.apply_along_axis()

    (3) Client-related arguments
    -----------------------------
    force_local:
    max_task_size: in Gb
    num_cores_per_worker:
    num_workers:
    return_done:
    compute_kwargs:

    (4) Architecture of computation
    --------------------------------
    loop_nchunk: number of chunks on which dask will create a task graph
    reduce_task_graph: whether or not to treat each index as a task

    """

    print(f"Received mappable of type {type(mappable)}")

    # (1) Determine chunks where separate task graphs will be computed

    # 1) List/Array of indices
    # 2) List/Array of strings (e.g. file paths)
    # 3) List of dask arrays / futures
    if isinstance(mappable, list) or isinstance(mappable, np.ndarray):

        if loop_nchunk is not None:

            from microbiome._vendor.analysis_toolbox.utils import chunk_list

            mappable = chunk_list(mappable, loop_nchunk)

        if mappable_id == "index":
            if isinstance(mappable, list):

                from microbiome._vendor.analysis_toolbox.utils import count_number_of_nested_lists

                number_of_nested_lists = count_number_of_nested_lists(mappable)

                if number_of_nested_lists < 2:
                    mappable = [mappable]

            # if it is not already in list form, make it a list
            if isinstance(mappable, np.ndarray):

                mappable = [mappable]

        elif isinstance(mappable, np.ndarray) and mappable_id == "data":

            mappable = [mappable]

    # 1) Typically array to be computed
    elif isinstance(mappable, da.Array) or isinstance(mappable, xr.DataArray):

        print(f"Single object detected...!")
        # Possibly a dask array
        mappable = [mappable]

    else:

        print(f"Not sure what, but assuming it's a single object")
        mappable = [mappable]

    from microbiome._vendor.analysis_toolbox.parallel import start_cluster
    import distributed
    from distributed import Client
    import time

    # (2) Determine number of workers
    if force_local:  # only 1 worker
        if max_task_size is not None:
            cores = num_cores_per_worker
            processes = calculate_num_processes(max_task_size, cores)
        else:
            print(f"Task size not provided, using all cores...")
            cores = num_cores_per_worker
            processes = (
                num_workers  # os.cpu_count() // cores  # use all cores on workstation
            )

        print(f"Number of workers: {processes}")
        print(f"Processes per worker: {cores}")

    else:
        cores = num_cores_per_worker
        processes = 1  # 1 process per worker

        print(f"Number of workers: {num_workers}")
        print(f"Processes per worker: {processes}")
        print(f"Cores per worker: {cores}")

    with start_cluster(
        cores=cores,
        force_local=force_local,
        processes=processes,
        death_timeout=1200,
        store_logs=True,
        spillover=True,
        verbose=True,
        # worker_lifetime="40m",
        # worker_lifetime_stagger="2m",
    ) as cluster, Client(cluster) as client:

        display(client)

        if not type(cluster) == distributed.deploy.local.LocalCluster:
            assert num_workers is not None
            scl_start = time.process_time()
            print(f"LSFCluster started! Scaling up to {num_workers} jobs")
            cluster.scale(num_workers)
            client.wait_for_workers(num_workers)
            print(f"Scaling time: {time.process_time() - scl_start} s")
        else:
            print(f"LocalCluster started!")

        results = []
        dones = []

        print(f"Computing...")
        print(f"(showing the first 10...)")
        display(mappable[:10])

        for mappable_ in tqdm(mappable):

            result, done = dask_client_schedule_and_compute(
                client,
                mappable_,
                mappable_id=mappable_id,
                func=func,
                func_args=func_args,
                func_kwargs=func_kwargs,
                apply_along_axis_kwargs=apply_along_axis_kwargs,
                reduce_task_graph=reduce_task_graph,
                compute_kwargs=compute_kwargs,
                zarr_path=zarr_path,
            )

            results.append(result)
            dones.append(done)

        if not type(cluster) == distributed.deploy.local.LocalCluster:
            try:
                cluster.scale(0)
                client.shutdown()
            except:
                pass

    # results is always going to be a list
    if loop_nchunk is not None and isinstance(mappable, list):

        from microbiome._vendor.analysis_toolbox.utils import flatten_list_of_lists

        results = flatten_list_of_lists(results)

    elif reduce_task_graph:

        from microbiome._vendor.analysis_toolbox.utils import flatten_list_of_lists

        results = flatten_list_of_lists(flatten_list_of_lists(results))

    elif len(results) == 1:

        results = results[0]

    if return_done:
        return results, dones
    else:
        return results


def dask_client_schedule_and_compute(
    client,
    mappable,
    mappable_id: str = "index",
    func: Callable = None,
    func_args: tuple = (),
    func_kwargs: dict = {},
    apply_along_axis_kwargs: dict = {"axis": 1, "dtype": None, "shape": None},
    reduce_task_graph: bool = False,
    compute_kwargs: dict = {},
    zarr_path: str = "",
):
    """
    func: map if a function is provided, otherwise, just compute
    func_args: for da.apply_along_axis and client.map
    func_kwargs: for da.apply_along_axis and client.map
    apply_along_axis_kwargs [dict]:
    """

    import time

    sch_start = time.process_time()

    if mappable_id == "data":

        # If func is provided, apply the function
        # Apply func1d, so only allowing 2-dim arrays
        # if function is a string, then I've defined it to call a method
        if func is not None:

            if callable(func):

                if isinstance(mappable, np.ndarray):

                    from microbiome._vendor.analysis_toolbox.fileio import chunk_and_saveas_zarr

                    nchunks = (
                        np.sum(list(client.nthreads().values())) * 4
                    )  # expected: cores*processes*4

                    print(f"nchunks: {nchunks}")

                    chunklen = np.clip(mappable.shape[0] // nchunks, 1, len(mappable))

                    # TODO: might require chunk size rather than nchunks
                    mappable_zarr, mappable_da = chunk_and_saveas_zarr(
                        mappable, debug=True, chunks=(chunklen, -1), reload="dask"
                    )

                elif isinstance(mappable, da.Array) or isinstance(
                    mappable, xr.DataArray
                ):

                    mappable_da = mappable

                display(mappable_da)

                future = da.apply_along_axis(
                    func,
                    apply_along_axis_kwargs["axis"],
                    mappable_da,
                    *func_args,
                    dtype=apply_along_axis_kwargs["dtype"],
                    shape=apply_along_axis_kwargs["shape"],
                    **func_kwargs,
                )

                display(future)

                if zarr_path:
                    from microbiome._vendor.analysis_toolbox.fileio import dask_arr_to_zarr_future

                    future = dask_arr_to_zarr_future(future, zarr_path, mode="da")
                    display(future)

                future = client.compute(future, **compute_kwargs)

            # if func provided is not callable
            else:

                future = None

        # func is not provided
        # it is probably already the result,
        # so just compute it
        else:
            display(mappable)

            from distributed import Future

            if (
                isinstance(mappable, da.Array)
                or isinstance(mappable, Future)
                or isinstance(mappable, xr.DataArray)
                or isinstance(mappable, db.Bag)
            ):

                if zarr_path:
                    from microbiome._vendor.analysis_toolbox.fileio import dask_arr_to_zarr_future

                    mappable = dask_arr_to_zarr_future(mappable, zarr_path, mode="da")
                    display(mappable)

                future = client.compute(mappable, **compute_kwargs)

            else:

                future = None
                print(f"What do you want me to do with this {type(mappable)}?!")

    # implement for loop version for results with huge task graphs
    elif mappable_id == "index":

        if func is not None:

            if callable(func):

                if reduce_task_graph:

                    # define a function to take in a list of inputs
                    # apply the func to elements of that list
                    def func_(list_of_inputs, **func_kwargs):

                        return [
                            func(input_, **func_kwargs) for input_ in list_of_inputs
                        ]

                    nchunks = (
                        np.sum(list(client.nthreads().values())) * 4
                    )  # expected: cores*processes*4

                    future = client.map(
                        func_, np.array_split(mappable, nchunks), **func_kwargs
                    )

                else:

                    future = client.map(func, mappable, **func_kwargs, **compute_kwargs)

            elif isinstance(func, str):

                future = None

        else:
            future = None
            print("What do you want me to do with these indices?!")

    print(f"Scheduling time: {time.process_time() - sch_start} s")

    cmp_start = time.process_time()
    if future is not None:
        result = client.gather(future)
    else:
        if isinstance(func, str):
            result = getattr(mappable, func)()
        else:
            print("No work done...!")

    print(f"Compute time: {time.process_time() - cmp_start} s")

    if future is not None:

        if isinstance(future, list):
            done = np.all([fut.done() for fut in future])
        else:
            done = future.done()

        if done:
            print(f"Futures computed successfully!")
        else:
            print(f"Something went wrong...")

    else:
        done = None

    try:
        cluster.scale(0)
    except:
        pass

    return result, done


def parallelize(data, func, num_of_processes=None, verbose=False, **kwargs):
    """
    func applies to a block of data and should include a for loop to apply another
    function to each data point.
    """

    if num_of_processes is None:
        num_of_processes = num_cpu()

    if verbose:
        print(f"number of processes: {num_of_processes}")

    from multiprocessing import Pool

    # split data into certain number of processes
    if isinstance(data, np.ndarray):
        data_split = np.array_split(data, num_of_processes)
    elif isinstance(data, list):
        from microbiome._vendor.analysis_toolbox.utils import list_split

        data_split = list_split(data, num_of_processes)

    if verbose:
        if isinstance(data, np.ndarray):
            data_split_size = [arr.shape for arr in data_split]
        elif isinstance(data, list):
            data_split_size = [len(arr) for arr in data_split]

        print("Data splits:")
        print(data_split_size)

    # create pool
    pool = Pool(num_of_processes)

    from microbiome._vendor.analysis_toolbox.utils import concatenate_list_of_lists

    if kwargs:
        from functools import partial

        func = partial(func, **kwargs)

    result = concatenate_list_of_lists(pool.map(func, data_split))

    pool.close()
    pool.join()

    return result
