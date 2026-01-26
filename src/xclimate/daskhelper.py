"""
Dask cluster utility.
"""

from __future__ import annotations
from typing import Optional
import os
import time
import platform
from glob import glob

from dask_jobqueue import PBSCluster
from dask.distributed import Client, get_client


def is_dask_available() -> bool:
    """Check if a Dask cluster is running and accessible"""
    try:
        client = get_client()
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
        print("\nTo view the dasl dashboard")
        print("Run the following command in your local terminal:")
        print(
            f"> ssh -N -L {port}:{address}:{port} {user}@{node}.hpc.ucar.edu"
        )  # local command line argument
        print("Open the following link in your local browser:")
        print(f"> http://localhost:{port}/status")  # link to local dask dashboard

    return client, cluster


def close_dask_cluster(
    client,
    cluster,
    remove_std_files: Optional[bool] = True,
) -> None:
    """Close dask cluster and clean up the workspace."""
    client.close()
    cluster.close()
    if remove_std_files:
        for f in glob("dask-worker.*"):
            os.remove(f)
