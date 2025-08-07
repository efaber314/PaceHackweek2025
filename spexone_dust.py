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
import argparse
from datetime import datetime, timedelta

# --- Helper functions for data retrieval ---
def get_spexone_data(files_auto, target_var, chi2_max, region):
    """Read SPEXone L2 data and return lists of lon, lat, and masked data for the target variable."""
    all_lons, all_lats, all_data_masked = [], [], []
    lon_min, lon_max, lat_min, lat_max = region
    for f in files_auto:
        try:
            with h5py.File(f, mode="r") as ds:
                if target_var in ds.get('geophysical_data', {}):
                    aot550 = ds['geophysical_data']['aot550'][:]
                    data = ds['geophysical_data'][target_var][:]
                    lat = ds['geolocation_data']['latitude'][:]
                    lon = ds['geolocation_data']['longitude'][:]
                    chi2 = ds['diagnostic_data']['chi2'][:]
                    fill = ds['geophysical_data'][target_var].attrs['_FillValue']
                    region_mask = ((lon >= lon_min)&(lon <= lon_max)&(lat >= lat_min)&(lat <= lat_max))
                    masked = np.ma.masked_where(
                        (~region_mask) |
                        (aot550 == fill) | (aot550 < 0) | (aot550 > 5) |
                        np.isnan(aot550) | (chi2 > chi2_max),
                        data
                    )
                    if masked.count() > 0:
                        all_lons.append(lon)
                        all_lats.append(lat)
                        all_data_masked.append(masked)
        except Exception:
            continue
    return all_lons, all_lats, all_data_masked

def get_oci_chl_dataset(files_chl):
    """Open multiple OCI chlorophyll files into an xarray Dataset."""
    return xr.open_mfdataset(files_chl, combine="nested", concat_dim="date")

%matplotlib inline


# %%
auth = earthaccess.login("../login.netrc")

# %% [markdown]
# Open SPEXone L2 dataset including only ocean

# %%
region =  (-30, -12,  5, 30) #WESN
region_plot = (-30, -20,  5, 30)

region =  (-60, 0,  5, 20) #WESN
region_plot = region
#region_plot = (-30, -20,  5, 30)

target_var="angstrom_440_670"
target_var_label="AE(440nm, 670nm)"
#target_var="aot550"
#target_var_label="AOD(550nm)"
#target_var="ssa440"
#target_var_label="AOD(550nm)"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process SPEXone and OCI data for a 2-week period.")
parser.add_argument('--start_date', type=str, required=True,
                    help="Start date in YYYY-MM-DD format")
parser.add_argument('--with_chl', action='store_true',
                    help="Include chlorophyll data")
args = parser.parse_args()
# Compute two-week window
start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
end_dt = start_dt + timedelta(days=13)
tspan = (args.start_date, end_dt.strftime("%Y-%m-%d"))
with_chl = args.with_chl

chi2_max=10 # 10 for ocean, 5 for land

plot_title=f"{target_var_label}, {tspan[0]} to {tspan[1]}\n chi2_max = {chi2_max}"
# %%

results = earthaccess.search_data(
    short_name="PACE_SPEXONE_L2_AER_RTAPOCEAN",
    temporal = tspan,
    bounding_box = region 
)
files_auto = earthaccess.open(results)  # Auto-detects provider

# Aggregate SPEXone data
all_lons, all_lats, all_data_masked = get_spexone_data(files_auto, target_var, chi2_max, region)

if with_chl:
    results_chl = earthaccess.search_data(
        short_name="PACE_OCI_L3M_CHL",
        temporal=tspan,
        granule_name="*.DAY.*.0p1deg.*",
        bounding_box = region 
    )
    files_chl = earthaccess.open(results_chl) 

    # Load chlorophyll data
    ds_chl = get_oci_chl_dataset(files_chl)



print(h5py.File(files_auto[0], mode="r")['geophysical_data'].keys())

# %%

# --- Plotting aod map---
if all_lons:
    print("\n--- Creating composite plot ---")
    # Create a figure with three stacked map panels
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(20, 15),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Left panel: SPEXone AOD
    ax1.set_extent(region, crs=ccrs.PlateCarree())
    # Replace default blue ocean background with white
    ax1.add_feature(cfeature.OCEAN, facecolor='white', zorder=0, alpha=1.0)
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0, alpha=0.8)
    ax1.add_feature(cfeature.COASTLINE, zorder=1)
    gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl1.top_labels = False
    gl1.right_labels = False
    ax1.set_title(f"{target_var_label} ({tspan[0]} to {tspan[1]})\nchi2_max={chi2_max}")
    mesh1 = None
    for lon, lat, aot in zip(all_lons, all_lats, all_data_masked):
        mesh1 = ax1.pcolormesh(
            lon, lat, aot, cmap='hot_r',
            vmin=0.4, vmax=1.6, transform=ccrs.PlateCarree(), zorder=2
        )
    cbar1 = fig.colorbar(mesh1, ax=ax1, label=target_var_label)

    # Right panel: OCI Chlorophyll
    ds_chl_log = np.log10(ds_chl)
    chl_var = list(ds_chl_log.data_vars)[0]
    ds_chl_avg = ds_chl[chl_var].mean("date", keep_attrs=True)
    ax2.set_extent(region, crs=ccrs.PlateCarree())
    # Replace default blue ocean background with white
    ax2.add_feature(cfeature.OCEAN, facecolor='white', zorder=0, alpha=1.0)
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0, alpha=0.8)
    ax2.add_feature(cfeature.COASTLINE, zorder=1)
    gl2 = ax2.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
    mesh2 = ds_chl_avg.plot.pcolormesh(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        cmap='viridis', alpha=0.5,
        vmin=0, vmax=0.2,
        add_colorbar=False
    )
    cbar2 = fig.colorbar(mesh2, ax=ax2, label='Chlorophyll a')
    ax2.set_title(f"Chlorophyll a ({tspan[0]} to {tspan[1]})")

    # Bottom panel: overlay SPEXone AOD on Chlorophyll
    ax3.set_extent(region, crs=ccrs.PlateCarree())
    ax3.add_feature(cfeature.OCEAN, facecolor='white', zorder=0, alpha=1.0)
    ax3.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0, alpha=0.8)
    ax3.add_feature(cfeature.COASTLINE, zorder=1)
    gl3 = ax3.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl3.top_labels = False
    gl3.right_labels = False
    # Plot chlorophyll
    mesh_chl3 = ds_chl_avg.plot.pcolormesh(
        ax=ax3,
        transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        cmap='viridis', alpha=0.5,
        vmin=0, vmax=0.2,
        add_colorbar=False
    )
    cbar_chl3 = fig.colorbar(mesh_chl3, ax=ax3, label='Chlorophyll a ', shrink=0.7)
    # Overlay SPEXone AOD
    mesh_spex3 = None
    for lon, lat, aot in zip(all_lons, all_lats, all_data_masked):
        mesh_spex3 = ax3.pcolormesh(
            lon, lat, aot, cmap='hot_r',
            vmin=0.4, vmax=1.6, transform=ccrs.PlateCarree(), zorder=2, alpha=0.7
        )
    cbar_spex3 = fig.colorbar(mesh_spex3, ax=ax3, label=target_var_label, shrink=0.7)
    ax3.set_title(f"{target_var_label} over Chlorophyll a ({tspan[0]} to {tspan[1]})")

    plt.tight_layout()
    output_path = f"hackweek2025_plots/new_{target_var}_{tspan[0]}__{tspan[1]}_map_composite.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
else:
    print("\nNo valid data found in any file to plot.")



# %%
