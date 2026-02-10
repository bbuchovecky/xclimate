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
def load_member_id_dict(ppe="fhist") -> dict:
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
    member_id_map = load_member_id_dict(ppe)
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


@lru_cache(maxsize=4)
def load_member_cat_dict(ppe="fhist") -> dict:
    """Load a member category dictionary from the YAML file."""
    member_cat_path = PKG_DIR / f"dicts/{ppe}_categories.yml"
    with open(member_cat_path, "r") as f:
        member_cat = yaml.safe_load(f)
    return member_cat


def get_member_cat_name(
    member_id: int | float | str | List[int | float | str] | np.ndarray | xr.DataArray,
    ppe: str = "fhist",
) -> str | List[str]:
    """Get the parameter functional category for a given member."""
    cat = load_member_cat_dict(ppe)
    info = get_member_info(member_id, no_id=False, ppe=ppe)

    if isinstance(info, List):
        return [cat["pcat_abbrv"][cat["pcat_group_inv"][str(i[0])]] for i in info]
    return cat["pcat_abbrv"][cat["pcat_group_inv"][str(info[0])]]


def get_member_cat_color(
    member_id: int | float | str | List[int | float | str] | np.ndarray | xr.DataArray,
    ppe: str = "fhist",
) -> str | List[str]:
    """Get the color for the parameter functional category of a given member."""
    cat = load_member_cat_dict(ppe)
    info = get_member_info(member_id, no_id=False, ppe=ppe)

    if isinstance(info, List):
        abbrv = [(cat["pcat_group_inv"][str(i[0])]) for i in info]
        return [cat["pcat_color"][a] for a in abbrv]
    
    return cat["pcat_color"][cat["pcat_group_inv"][str(info[0])]]
    
