"""
Dask cluster utility.
"""

from __future__ import annotations
from typing import Optional, Tuple
import os
import time
import platform
from glob import glob

from dask_jobqueue import PBSCluster
from dask.distributed import Client, get_client


def is_dask_available() -> bool:
    """Check if a Dask cluster is running and accessible"""
    try:
        get_client()
        return True
    except ValueError:
        return False


def create_dask_cluster(
    account: str,
    nworkers: int,
    ncores: int = 1,
    nmem: str = "5GB",
    walltime: str = "01:00:00",
    print_dash: Optional[bool] = True,
    **kwargs,
):
    """
    Create and scale a dask cluster on either casper or derecho.
    https://ncar.github.io/dask-tutorial/notebooks/05-dask-hpc.html

    Parameters
    ----------
        account : str
            Account to charge core hours for dask workers.
        nworkers : int
            Number of workers to scale up.
        ncores : int
            Requested number of cores.
        nmem : str
            Requested amount of memory, in the form 'XGB'.
        walltime : str
            Requested walltime, in the form '00:00:00'.
        print_dash : Optional[bool]
            Whether to print instrcutions to access dask dashboard, defaults to True.
        **kwargs
            Arguments to pass to PBSCluster.

    Returns
    -------
    client, cluster
        Dask objects corresponding to the client and cluster.
    """
    node = platform.node()
    if "crlogin" in node:
        node = "casper"
        queue = "casper"
        interface = "ext"
    elif "derecho" in node:
        node = "derecho"
        queue = "develop"
        interface = "hsn0"
    else:
        raise KeyError(
            'must be on "casper" or "derecho", other machines not implemented'
        )

    # Print requested resources
    print(f"account:  {account}")
    print(f"nworkers: {nworkers}")
    print(f"ncores:   {ncores}")
    print(f"nmemory:  {nmem}")
    print(f"walltime: {walltime}")

    # Create the cluster and scale to size
    cluster = PBSCluster(
        cores=ncores,
        memory=nmem,
        queue=queue,
        interface=interface,
        resource_spec=f"select=1:ncpus={str(ncores)}:mem={nmem}",
        account=account,
        walltime=walltime,
        **kwargs,
    )
    client = Client(cluster)
    cluster.scale(nworkers)
    time.sleep(5)

    print(cluster.workers)

    # Create a SSH tunnel to access the dask dashboard locally
    if print_dash:
        user = os.environ.get("USER")
        port = cluster.dashboard_link.split(":")[2].split("/")[0]
        address = cluster.dashboard_link.split(":")[1][2:]
        print("\nTo view the dask dashboard")
        print("Run the following command in your local terminal:")
        print(
            f"> ssh -N -L {port}:{address}:{port} {user}@{node}.hpc.ucar.edu"
        )  # local command line argument
        print("Open the following link in your local browser:")
        print(f"> http://localhost:{port}/status")  # link to local dask dashboard

    return (client, cluster)


def close_dask_cluster(
    client_cluster: Tuple,
    remove_std_files: Optional[bool] = True,
) -> None:
    """Close dask cluster and clean up the workspace."""
    client, cluster = client_cluster
    client.close()
    cluster.close()
    if remove_std_files:
        for f in glob("dask-worker.*"):
            os.remove(f)


def get_ncpus(default=1) -> int:
    """
    Get the number of CPUs available from environment variables.
    
    Checks PBS_NCPUS, NCPUS, and OMP_NUM_THREADS environment variables
    in order, falling back to os.cpu_count() or the default value.

    Parameters
    ----------
        default : int
            Default number of CPUs to return if none can be detected.

    Returns
    -------
    int
        Number of available CPUs.
    """
    for k in ("PBS_NCPUS", "NCPUS", "OMP_NUM_THREADS"):
        v = os.environ.get(k)
        if v and v.isdigit():
            return int(v)
    return os.cpu_count() or default


def get_memory_per_worker(n_workers, overhead_fraction=0.1, default="4GB") -> str:
    """
    Calculate per-worker memory limit from PBS allocation.
    
    Checks PBS environment variables for total memory allocation and divides
    by number of workers, reserving a fraction for system overhead.

    Parameters
    ----------
        n_workers : int
            Number of Dask workers.
        overhead_fraction : float
            Fraction of memory to reserve for system/scheduler (default 10%).
        default : str
            Fallback memory limit if PBS memory not detected.

    Returns
    -------
    str
        Memory limit string (e.g., "7GB").
    """
    # Try to get total memory from PBS
    total_mem_kb = None
    for k in ("PBS_RESC_TOTAL_MEM", "PBS_VMEM", "PBS_MEM"):
        v = os.environ.get(k)
        if v:
            # Parse formats like "64gb", "64000mb", "65536000kb"
            v_lower = v.lower().strip()
            if v_lower.endswith("gb"):
                total_mem_kb = int(float(v_lower[:-2]) * 1024 * 1024)
            elif v_lower.endswith("mb"):
                total_mem_kb = int(float(v_lower[:-2]) * 1024)
            elif v_lower.endswith("kb"):
                total_mem_kb = int(float(v_lower[:-2]))
            if total_mem_kb:
                break
    
    if total_mem_kb:
        # Reserve overhead and divide by workers
        usable_mem_kb = int(total_mem_kb * (1 - overhead_fraction))
        per_worker_kb = usable_mem_kb // n_workers
        per_worker_gb = per_worker_kb / (1024 * 1024)
        return f"{per_worker_gb:.1f}GB"
    
    return default
