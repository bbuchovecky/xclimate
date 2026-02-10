#!/usr/bin/env python3
"""
Script to generate the parameter categories dict for the FHIST PPE.
"""

import yaml


pcat_abbrv = {
    "df": "default",
    "sl": "soil hydrology",
    "bl": "boundary layer and roughness",
    "wu": "plant water use",
    "ps": "photosynthesis",
    "ce": "canopy evaporation",
    "ta": "temperature acclimation",
}

pcat_group = {
    "df": ['0'],
    "sl": ['3', '4', '7', '8'],
    "bl": ['1', '2'],
    "wu": ['9', '10', '11', '12', '15', '16', '17', '18'],
    "ps": ['13', '14', '19', '20', '21', '22', '25', '26'],
    "ce": ['5', '6'],
    "ta": ['23', '24', '27', '28'],
}

pcat_group_inv = {param: key for key, params in pcat_group.items() for param in params}

pcat_color = {
    "df": "#5C5C5C",
    "sl": "#0069CC",
    "bl": "#75147C",
    "wu": "#458933",
    "ps": "#55AFA9",
    "ce": "#97CCE8",
    "ta": "#E69200",
}

pcat_data = {
    'pcat_abbrv': pcat_abbrv,
    'pcat_group': pcat_group,
    'pcat_group_inv': pcat_group_inv,
    'pcat_color': pcat_color,
}

with open('../dicts/fhist_categories.yml', 'w') as f:
    yaml.dump(pcat_data, f, default_flow_style=False)
