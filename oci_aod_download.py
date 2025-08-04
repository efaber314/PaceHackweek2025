from pathlib import Path

import cartopy.crs as ccrs
import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import requests, pdb
import xarray as xr

auth = earthaccess.login(persist=True)
fs = earthaccess.get_fsspec_https_session()

tspan = ("2024-07-22", "2024-09-30")

bbox = (-65.44, 8.03, -5.06, 38.75) # W, S, E, N
clouds = (0, 100) #cloud cover 0-50%

#===============================================
#1. get lists of timestamps that sweeps thru our coordinates

#2. make a for loop for all the timestamps and download oci aod (because oci aod is not offered in earthaccess, we cannot designate bbox)
#===============================================

#1.

results = earthaccess.search_data(
    short_name="PACE_OCI_L2_BGC", #example var.
    temporal=tspan,
    bounding_box=bbox,
    cloud_cover=clouds,
)
# (9 24 25)
tstamps = [] #lists of timestamps
for i in range(len(results)):
    fname = results[i]['meta']['native-id']
    rec = fname.strip().split('.')
    tstamps.append(rec[1])
    
#2.
OB_DAAC_PROVISIONAL = "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"
for tstamp in tstamps:
    
    HARP2_L2_MAPOL_FILENAME = "PACE_OCI.{}.L2.AER_UAA.V3_0.NRT.nc".format(tstamp)
    try:
        fs.get(f"{OB_DAAC_PROVISIONAL}/{HARP2_L2_MAPOL_FILENAME}", "/home/jovyan/shared-public/DustBlumes/PACE_OCI_AOD/")
        print(tstamp)
    except:
        pass
    # paths = list(Path("data").glob("*.nc"))
    # for path in paths:
    #     reader = [
    # pdb.set_trace()