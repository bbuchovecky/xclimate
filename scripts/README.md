# Scripts

## generate_variables_dict.py

This script scans a CESM timeseries directory and automatically generates a VARIABLES dictionary compatible with the `fhist_variables.py` format.

### Usage

```bash
python generate_variables_dict.py <tseries_directory> [options]
```

### Arguments

- `tseries_dir`: Path to the timeseries directory containing frequency subdirectories (e.g., `month_1`, `day_1`)

### Options

- `-o, --output`: Output file path (default: print to stdout)
- `--stats`: Print statistics about discovered variables

### Examples

**Print to stdout:**
```bash
python generate_variables_dict.py /glade/campaign/univ/uwas0155/ppe/historical/coupled_simulations/f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE.003/atm/proc/tseries
```

**Save to file with statistics:**
```bash
python generate_variables_dict.py \
  /glade/campaign/univ/uwas0155/ppe/historical/coupled_simulations/f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE.003/atm/proc/tseries \
  -o my_variables.py \
  --stats
```

### How it works

1. Scans all frequency subdirectories within the tseries directory
2. Parses each `.nc` filename to extract:
   - Variable name
   - Stream (h0, h1, h2, h3)
   - Component (cam → atm, clm2 → lnd)
   - Frequency (month_1, day_1, etc.)
3. Generates a Python dictionary with the format:
   ```python
   VARIABLES = {
       "TREFHT_month_1": Variable("TREFHT", "h0", "atm", "month_1"),
       ...
   }
   ```

### Expected filename format

The script expects timeseries files with the naming pattern:
```
<case_name>.<component>.<stream>.<variable>.<date_range>.nc
```

Example:
```
f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE.003.cam.h0.TREFHT.195001-201412.nc
```

### Output format

The generated output can be directly used in `fhist_variables.py` or any other script that uses the `Variable` dataclass format.
