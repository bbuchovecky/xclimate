#!/usr/bin/env python3
"""
Script to parse timeseries files and generate a VARIABLES dictionary.

This script scans a tseries directory (with frequency subdirectories like month_1, day_1),
parses the timeseries filenames, and creates a VARIABLES dictionary compatible with
the fhist_variables.py format.

Usage:
    python generate_variables_dict.py <tseries_directory>

Example:
    python generate_variables_dict.py /glade/campaign/univ/uwas0155/ppe/historical/coupled_simulations/f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE.003/atm/proc/tseries
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set


def parse_filename(filename: str) -> dict | None:
    """
    Parse a tseries filename and extract metadata.
    
    Expected format:
    <case_name>.<component>.<stream>.<variable>.<date_range>.nc
    
    Example:
    f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE.003.cam.h0.TREFHT.195001-201412.nc
    
    Returns:
        dict with keys: component, stream, variable
        None if parsing fails
    """
    # Pattern to match the filename structure
    # We're looking for: <anything>.<comp>.<stream>.<variable>.<dates>.nc
    pattern = r'^.+\.(cam|clm2)\.([hH][0-9])\.([^.]+)\.\d+-\d+\.nc$'
    
    match = re.match(pattern, filename)
    if not match:
        return None
    
    component = match.group(1)
    stream = match.group(2).lower()  # Normalize to lowercase
    variable = match.group(3)
    
    return {
        'component': component,
        'stream': stream,
        'variable': variable
    }


def get_general_component(specific_comp: str) -> str:
    """Convert specific component (cam, clm2) to general component (atm, lnd)."""
    comp_map = {
        'cam': 'atm',
        'clm2': 'lnd'
    }
    return comp_map.get(specific_comp, specific_comp)


def scan_tseries_directory(tseries_dir: Path) -> Dict[str, dict]:
    """
    Scan tseries directory and extract variable metadata.
    
    Args:
        tseries_dir: Path to the tseries directory
        
    Returns:
        Dictionary mapping variable keys to metadata
    """
    variables = {}
    
    # Scan all frequency subdirectories
    for freq_dir in tseries_dir.iterdir():
        if not freq_dir.is_dir():
            continue
        
        frequency = freq_dir.name  # e.g., 'month_1', 'day_1'
        
        # Scan all .nc files in this frequency directory
        for nc_file in freq_dir.glob('*.nc'):
            parsed = parse_filename(nc_file.name)
            
            if parsed is None:
                continue
            
            variable_name = parsed['variable']
            stream = parsed['stream']
            specific_comp = parsed['component']
            general_comp = get_general_component(specific_comp)
            
            # Create a unique key for this variable
            var_key = f"{variable_name}_{frequency}"
            
            # Store metadata
            variables[var_key] = {
                'name': variable_name,
                'stream': stream,
                'gcomp': general_comp,
                'frequency': frequency
            }
    
    return variables


def generate_python_code(variables: Dict[str, dict]) -> str:
    """
    Generate Python code for the VARIABLES dictionary.
    
    Args:
        variables: Dictionary of variable metadata
        
    Returns:
        Python code as a string
    """
    # Sort variables by key for consistent output
    sorted_vars = sorted(variables.items())
    
    # Group by component for better organization
    atm_vars = [(k, v) for k, v in sorted_vars if v['gcomp'] == 'atm']
    lnd_vars = [(k, v) for k, v in sorted_vars if v['gcomp'] == 'lnd']
    
    lines = ['VARIABLES = {']
    
    if atm_vars:
        lines.append('    # CAM variables')
        for var_key, var_data in atm_vars:
            line = f'    "{var_key}": Variable("{var_data["name"]}", "{var_data["stream"]}", "{var_data["gcomp"]}", "{var_data["frequency"]}"),'
            lines.append(line)
    
    if lnd_vars:
        if atm_vars:
            lines.append('')
        lines.append('    # CLM variables')
        for var_key, var_data in lnd_vars:
            line = f'    "{var_key}": Variable("{var_data["name"]}", "{var_data["stream"]}", "{var_data["gcomp"]}", "{var_data["frequency"]}"),'
            lines.append(line)
    
    lines.append('}')
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate VARIABLES dictionary from tseries directory'
    )
    parser.add_argument(
        'tseries_dir',
        type=str,
        help='Path to the tseries directory'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (default: print to stdout)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print statistics about discovered variables'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    tseries_path = Path(args.tseries_dir)
    if not tseries_path.exists():
        print(f"Error: Directory does not exist: {tseries_path}")
        return 1
    
    if not tseries_path.is_dir():
        print(f"Error: Not a directory: {tseries_path}")
        return 1
    
    # Scan directory
    print(f"Scanning directory: {tseries_path}", flush=True)
    variables = scan_tseries_directory(tseries_path)
    
    if not variables:
        print("Warning: No variables found in the directory")
        return 1
    
    # Print statistics if requested
    if args.stats:
        print(f"\nFound {len(variables)} variables:")
        
        # Count by component
        comp_counts = defaultdict(int)
        freq_counts = defaultdict(int)
        stream_counts = defaultdict(int)
        
        for var_data in variables.values():
            comp_counts[var_data['gcomp']] += 1
            freq_counts[var_data['frequency']] += 1
            stream_counts[var_data['stream']] += 1
        
        print("\nBy component:")
        for comp, count in sorted(comp_counts.items()):
            print(f"  {comp}: {count}")
        
        print("\nBy frequency:")
        for freq, count in sorted(freq_counts.items()):
            print(f"  {freq}: {count}")
        
        print("\nBy stream:")
        for stream, count in sorted(stream_counts.items()):
            print(f"  {stream}: {count}")
        
        print()
    
    # Generate code
    code = generate_python_code(variables)
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(code)
        print(f"Written to: {output_path}")
    else:
        print(code)
    
    return 0


if __name__ == '__main__':
    exit(main())
