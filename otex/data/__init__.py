# -*- coding: utf-8 -*-
"""
OTEX Data Module
Oceanographic data processing (CMEMS, NetCDF).
"""

from .cmems import (
    download_data,
    data_processing,
    load_temperatures,
)

__all__ = [
    "download_data",
    "data_processing",
    "load_temperatures",
]
