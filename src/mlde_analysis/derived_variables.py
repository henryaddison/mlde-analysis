"""
Module for working with variables that are computed from others.
For example, saturated wet globe bulb temperature can be estimated from temperature and relative humidity.

Resouces:
* https://github.com/OpenCLIM/HEAT-stress/blob/main/hurs2VP.m
* https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
* http://www.bom.gov.au/info/thermal_stress/#approximation
* https://iopscience.iop.org/article/10.1088/1748-9326/10/8/084013 - Zhao, Y., A. Ducharne, B. Sultan, P. Braconnot and R. Vautard (2015). "Estimating heat stress from climate-based indicators: present-day biases and future spreads in the CMIP5 global climate model ensemble." Environmental Research Letters 10(8): 084013.
* https://link.springer.com/article/10.1007/s00484-011-0453-2#Sec2 - Blazejczyk, K., Y. Epstein, G. Jendritzky, H. Staiger and B. Tinz (2012). "Comparison of UTCI to selected thermal indices." International Journal of Biometeorology 56(3): 515-535.
* https://gmd.copernicus.org/articles/8/151/2015/ - Buzan, J. R., Oleson, K., and Huber, M.: Implementation and comparison of a suite of heat stress metrics within the Community Land Model version 4.5, Geosci. Model Dev., 8, 151–170, https://doi.org/10.5194/gmd-8-151-2015, 2015.
"""

import numpy as np
import xarray as xr


def _k2c(temp: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius.

    Args:
        temp: temperature in Kelvin.

    Returns:
        temperature in Celsius.
    """
    return temp - 273.15


def _c2k(temp: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Celsius to Kelvin.

    Args:
        temp: temperature in Celsius.

    Returns:
        temperature in Kelvin.
    """
    return temp + 273.15


def vp(temp: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    """Compute the vapour pressure.


    Args:
        temp: temperature in Kelvin.
        rh: relative humidity as a percentage.

    Returns:
        Vapour pressure in hPa.
    """
    tempc = _k2c(temp)
    # Calculate saturated vapour pressure
    svp = 6.112 * np.exp((17.67 * tempc) / (tempc + 243.5))
    # an alternative is
    # svp = 6.105 * exp((17.27 * tempc) / (tempc + 237.7))
    #  Calculate vapour pressure
    return (rh / 100.0) * svp


def swbgt(temp: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    """Compute the simplified wet bulb globe temperature.

    Args:
        temp: temperature in Kelvin.
        rh: relative humidity as a percentage.

    Returns:
        simplified wet globe bulb temperature (Celsius).
    """
    temp_c = _k2c(temp)
    return (
        (0.567 * temp_c + 0.393 * vp(temp, rh) + 3.94)
        .rename("swgbt")
        .assign_attrs(
            {"long_name": "simplified Wet Bulb Globe Temperature", "units": "C"}
        )
    )


def apt(temp: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    """
    Compute the apparent temp from temperature and relative humidity.

    Args:
        temp: temperature in Kelvin.
        rh: relative humidity as a percentage.

    Returns:
        apparent temperature in Kelvin.
    """
    temp_c = _k2c(temp)
    atp_c = 0.92 * temp_c + 0.22 * vp(temp, rh) - 1.3

    return _c2k(atp_c).rename("atp")


def humidex(temp: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    """
    Compute the humidex from temperature and relative humidity.

    Args:
        temp: temperature in Kelvin.
        rh: relative humidity as a percentage.

    Returns:
        humidex in Kelvin.
    """
    temp_c = _k2c(temp)
    humidex_c = temp_c + vp(temp, rh) * 0.555 - 5.5

    return (
        _c2k(humidex_c)
        .rename("humidex")
        .assign_attrs(
            {
                "long_name": "Humidex",
                "units": "K",
                "units_metadata": "temperature: onscale",
            }
        )
    )
