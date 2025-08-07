# %%
import earthaccess
import h5py
import numpy as np
import pandas as pd
import matplotlib
# Set the backend to 'Agg' for non-interactive plotting, which is more reliable for scripts.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ssl
import os
import xarray as xr
import multiprocessing
import glob
from datetime import datetime, timedelta

# Maximum allowable chi2 for quality filtering
chi2_max = 10

# Define variables to extract and whether they have a wavelength dimension
spexone_vars = {
    "aot550": False,
    "angstrom_440_670": False,
    "aot_mode1": True,
    "aot_mode2": True,
    "aot_mode3": True,
    "chla" : False
}

%matplotlib inline

"""
"TAU_MODE1_550nm": "TAU_OBS_MODE_FINE_550nm",
"TAU_MODE2_550nm": "TAU_OBS_MODE_CI_550nm",
"TAU_MODE3_550nm": "TAU_OBS_MODE_CS_550nm",
7 is 550nm
"""
# %%
auth = earthaccess.login("../login.netrc")

import pandas as pd
from datetime import timedelta

# Define weekly spans
start_date = pd.to_datetime("2024-08-01")
end_date = pd.to_datetime("2024-11-01")
week_starts = pd.date_range(start=start_date, end=end_date, freq='7D')
weekly_spans = [(d.strftime("%Y-%m-%d"), (d + timedelta(days=6)).strftime("%Y-%m-%d")) for d in week_starts]

# Define region and plotting flag
region = (-60, 0, 5, 20)
region_plot = region
with_chl = True
chi2_max = 10
# %%

def read_oci(date_start, date_end):
    # Convert inputs to datetime
    if isinstance(date_start, str):
        date_start = datetime.strptime(date_start, "%Y-%m-%d")
    if isinstance(date_end, str):
        date_end = datetime.strptime(date_end, "%Y-%m-%d")

    oci_dir = '/home/jovyan/shared-public/DustBlumes/PACE_OCI_AOD/'
    aods, ssas, lons, lats = [], [], [], []

    current_date = date_start
    while current_date <= date_end:
        pattern = os.path.join(oci_dir, current_date.strftime('*%Y%m%dT*'))
        files = sorted(glob.glob(pattern))
        for file in files:
            datatree = xr.open_datatree(file)
            dataset = xr.merge(datatree.to_dict().values())
            lon = np.array(dataset['longitude'])
            lat = np.array(dataset['latitude'])
            # Aerosol Optical Depth at channel index 3
            aod = np.squeeze(np.array(dataset['Aerosol_Optical_Depth'][:,:,3]))
            # Single Scattering Albedo NUV channel index 1
            ssa = np.squeeze(np.array(dataset['NUV_AerosolSingleScattAlbedo'][:,:,1]))

            lons.append(lon)
            lats.append(lat)
            aods.append(aod)
            ssas.append(ssa)
        current_date += timedelta(days=1)
        
        
    # Flatten lists of 2D arrays into 1D arrays
    aods = np.concatenate([arr.flatten() for arr in aods]) if aods else np.array([])
    ssas = np.concatenate([arr.flatten() for arr in ssas]) if ssas else np.array([])
    lons = np.concatenate([arr.flatten() for arr in lons]) if lons else np.array([])
    lats = np.concatenate([arr.flatten() for arr in lats]) if lats else np.array([])
    # Filter points to the specified region
    lon_min, lon_max, lat_min, lat_max = region
    mask = (
        (lons >= lon_min) & (lons <= lon_max) &
        (lats >= lat_min) & (lats <= lat_max)
    )
    aods = aods[mask]
    ssas = ssas[mask]
    lons = lons[mask]
    lats = lats[mask]
    return aods, ssas, lons, lats

def read_spexone_span(tspan):
    """Read SPEXone aerosol metrics for the given time span and return a list of means in the order of spexone_vars."""
    week_data = {var: [] for var in spexone_vars}
    results = earthaccess.search_data(
        short_name="PACE_SPEXONE_L2_AER_RTAPOCEAN",
        temporal=tspan,
        bounding_box=region
    )
    files_auto = earthaccess.open(results)
    for f in files_auto:
        with h5py.File(f, mode="r") as ds_h5py:
            chi2 = ds_h5py['diagnostic_data']['chi2'][:]
            lon_arr = np.array(ds_h5py['geolocation_data']['longitude'])
            lat_arr = np.array(ds_h5py['geolocation_data']['latitude'])
            lon_min, lon_max, lat_min, lat_max = region
            region_mask = (
                (lon_arr >= lon_min) & (lon_arr <= lon_max) &
                (lat_arr >= lat_min) & (lat_arr <= lat_max)
            )
            aot550 = ds_h5py['geophysical_data']['aot550']
            for var, is_wavelength in spexone_vars.items():
                data = ds_h5py['geophysical_data'][var]
                fill = data.attrs['_FillValue']
                data_arr = data[:,:,7] if is_wavelength else data[:]
                mask = np.ma.masked_where(
                    (data_arr == fill) |
                    np.isnan(data_arr) |
                    (aot550[:] == aot550.attrs['_FillValue']) |
                    (aot550[:] < 0) |
                    (aot550[:] > 5) |
                    (chi2 > chi2_max) |
                    (~region_mask),
                    data_arr
                )
                if mask.count() > 0:
                    week_data[var].append(mask.mean())
    # Compute SPEXone means in order
    return [np.nanmean(week_data[var]) if week_data[var] else np.nan for var in spexone_vars]

def read_chl_span(tspan):
    """Read chlorophyll data for the given time span and return log10 mean."""
    
    """
    results_chl = earthaccess.search_data(
        short_name="PACE_OCI_L2_BGC",
        temporal=tspan,
        bounding_box=region
    )
    files_chl = earthaccess.open(results_chl)
    chls = []
    for file in files_chl:
        datatree = xr.open_datatree(file)
        dataset = xr.merge(datatree.to_dict().values())
        lon_arr = np.array(dataset['longitude'])
        lat_arr = np.array(dataset['latitude'])
        lon_min, lon_max, lat_min, lat_max = region
        region_mask = (
            (lon_arr >= lon_min) & (lon_arr <= lon_max) &
            (lat_arr >= lat_min) & (lat_arr <= lat_max)
        )
        chl = np.squeeze(np.array(dataset["chlor_a"]))
        chl = np.where(region_mask, chl, np.nan)
        chls.append(chl)
    chls_flat = np.concatenate([arr.flatten() for arr in chls]) if chls else np.array([])
    return np.nanmean(chls_flat) if chls_flat.size else np.nan
    """
    results_chl = earthaccess.search_data(
        short_name="PACE_OCI_L3M_CHL",
        temporal=tspan,
        bounding_box=region
    )
    files_chl = earthaccess.open(results_chl)
    ds_chl = xr.open_mfdataset(files_chl,
        combine="nested",
        concat_dim="date")["chlor_a"]
    # Filter chlorophyll to the specified region
    lon_arr = ds_chl["lon"]
    lat_arr = ds_chl["lat"]
    lon_min, lon_max, lat_min, lat_max = region
    ds_chl = ds_chl.where(
        (lon_arr >= lon_min) & (lon_arr <= lon_max) &
        (lat_arr >= lat_min) & (lat_arr <= lat_max),
        drop=True
    )
    return ds_chl.mean().compute()

# Process weekly spans in parallel using multiprocessing
def process_span(tspan):
    # Debug: print current process
    import os
    print(f"Process {multiprocessing.current_process().name} (PID {os.getpid()}) handling span {tspan}")
    # Convert start date
    week_start = pd.to_datetime(tspan[0])

    spexone_means = read_spexone_span(tspan)
    
    print(f"SPEXone done, {multiprocessing.current_process().name} (PID {os.getpid()}) handling span {tspan}")
   
    print(f"chla start, {multiprocessing.current_process().name} (PID {os.getpid()}) handling span {tspan}")
    
    chl_mean = read_chl_span(tspan)
    
    print(f"chla done, {multiprocessing.current_process().name} (PID {os.getpid()}) handling span {tspan}")
    
    """
    # Read OCI AOD data for this span
    aods, ssas, lons, lats = read_oci(tspan[0], tspan[1])
    print(f"aod done, {multiprocessing.current_process().name} (PID {os.getpid()}) handling span {tspan}")
    #aod_mean = np.nanmean(aods) if aods.size else np.nan
    """
    aod_mean = 0
    
    return week_start, *spexone_means, chl_mean, aod_mean

# Run in parallel
with multiprocessing.Pool() as pool:
    results = pool.map(process_span, weekly_spans)

unpacked = list(zip(*results))
weeks = list(unpacked[0])
var_means = unpacked[1:-1]
chl_means = list(unpacked[-2])
aod_means = list(unpacked[-1])

# Assign individual variable lists
var_mean_lists = {var: list(var_means[i]) for i, var in enumerate(spexone_vars)}

# %%
# %%
"""
# Plot time series

plt.figure(figsize=(10,6))
plt.plot(weeks, var_mean_lists["angstrom_440_670"], label="AE(440nm, 670nm)")
plt.plot(weeks, var_mean_lists["aot550"], label="AOD(550nm)")
plt.plot(weeks, var_mean_lists["aot_mode1"], label="AOT Mode1")
plt.plot(weeks, var_mean_lists["aot_mode2"], label="AOT Mode2")
plt.plot(weeks, var_mean_lists["aot_mode3"], label="AOT Mode3")
plt.xlabel("Week Starting")
plt.ylabel("Mean Value")
plt.title("Weekly Mean Aerosol Metrics")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(weeks, chl_means, label="Mean Log10 Chlorophyll")
plt.xlabel("Week Starting")
plt.ylabel("Mean Log10 Chlorophyll")
plt.title("Weekly Mean Chlorophyll")
plt.legend()
plt.grid(True)
plt.show()

fig, ax1 = plt.subplots(figsize=(10,6))

# left‐hand axis: AOT Mode2 in red
line1, = ax1.plot(weeks, var_mean_lists["aot_mode2"], color='red', label='AOT Mode2')
ax1.set_xlabel('Week Starting')
ax1.set_ylabel('AOT Mode2', color='red')
ax1.tick_params(axis='y', colors='red')
ax1.legend(loc='upper left')
ax1.grid(True)

# right‐hand axis: Mean Chlorophyll a
ax2 = ax1.twinx()
ax2.plot(weeks, chl_means, label='Mean Chlorophyll a')
ax2.set_ylabel('Mean Chlorophyll a')
ax2.legend(loc='upper right')

plt.title('AOT Mode2 (left, red) and Chlorophyll a (right)')
plt.tight_layout()
plt.show()
"""

# Dual-axis: OCI AOD (550nm) and Chlorophyll a
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(weeks, aod_means, label='OCI AOD (550nm)', color='blue')
ax1.set_xlabel('Week Starting')
ax1.set_ylabel('OCI AOD (550nm)', color='blue')
ax1.tick_params(axis='y', colors='blue')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(weeks, chl_means, label='Mean Chlorophyll a', color='green')
ax2.set_ylabel('Mean Chlorophyll a', color='green')
ax2.legend(loc='upper right')

plt.title('OCI AOD (left, blue) and Chlorophyll a (right)')
plt.tight_layout()
plt.show()


# Dual-axis: SPEXone AOD (550nm) and Chlorophyll a
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(weeks, var_mean_lists["aot550"], label='SPEXone AOD (550nm)', color='purple')
ax1.set_xlabel('Week Starting')
ax1.set_ylabel('SPEXone AOD (550nm)', color='purple')
ax1.tick_params(axis='y', colors='purple')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(weeks, var_mean_lists["chla"], label='SPEXone Mean Chlorophyll a', color='orange')
ax2.plot(weeks, chl_means, label='OCI Mean Chlorophyll a', color='green')
ax2.set_ylabel('Mean Chlorophyll a')
ax2.legend(loc='upper right')

plt.title('SPEXone AOD (left, purple) and Chlorophyll a (right)')
plt.tight_layout()
plt.show()
 # %%
