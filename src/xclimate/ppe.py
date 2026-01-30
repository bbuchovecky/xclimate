"""PPE utilities."""

from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple, Optional
from pathlib import Path
import yaml

import numpy as np
import xarray as xr

PKG_DIR = Path(__file__).parent.parent.parent


@lru_cache(maxsize=4)
def load_member_id_map(ppe="fhist") -> dict:
    """Load a member map dictionary from the YAML file."""
    member_id_map_path = PKG_DIR / f"dicts/{ppe}_members.yml"
    with open(member_id_map_path, "r") as f:
        member_id_map = yaml.safe_load(f)
    return member_id_map


def invert_member_id_map(d):
    inverted = {}
    for param, minmax_dict in d.items():
        for minmax, mem_id in minmax_dict.items():
            inverted[int(mem_id)] = (mem_id, param, minmax)
    return inverted


def get_member_info(
    member_id: int | float | str | List[int | float | str] | np.ndarray | xr.DataArray,
    no_id: Optional[bool] = False,
    ppe: str = "fhist",
) -> Tuple | List[Tuple]:
    """Get the tuple (member_id, parameter_name, minmax)."""
    member_id_map = load_member_id_map(ppe)
    inverted = invert_member_id_map(member_id_map)

    # Convert all inputs to list
    if isinstance(member_id, (int, float, str, np.floating, np.integer)):
        member_id = [member_id]
    elif isinstance(member_id, xr.DataArray):
        member_id = member_id.values.flatten()
    elif isinstance(member_id, np.ndarray):
        member_id = member_id.flatten()

    # Ensure list elements are appropriate type
    member_id = [
        int(m) if isinstance(m, (float, np.floating, np.integer)) else m for m in member_id
    ]

    info = []
    for mem_id in member_id:
        result = inverted.get(int(mem_id) if isinstance(mem_id, str) else mem_id)
        if result is not None:
            if no_id:
                info.append((result[1], result[2]))
            else:
                info.append(result)

    if len(info) == 1:
        return info[0]
    return info


def get_member_name(
    member_id: int | float | str | List[int | float | str] | np.ndarray | xr.DataArray,
    no_id: Optional[bool] = False,
    delimiter: str = ".",
    ppe: str = "fhist",
) -> str | List[str]:
    """Get a formatted member name string."""
    info = get_member_info(member_id, no_id, ppe)

    if isinstance(info, List):
        return [delimiter.join(str(x) for x in i) for i in info]
    return delimiter.join(str(x) for x in info)
