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

# This line creates an unverified SSL context, which helps bypass
# certificate verification errors that can occur when cartopy downloads
# map data on certain system configurations (like macOS).
ssl._create_default_https_context = ssl._create_unverified_context

# Log in to Earthdata
auth = earthaccess.login(strategy="netrc", persist=True)
# If no .netrc, use: auth = earthaccess.login(strategy="interactive")
print("Logged in:", auth.authenticated)

# -- number of the data products (total, non-cloud_hostead and cloud_hosted ) ---
# datasets = earthaccess.search_datasets(keyword=" ")
# print(f"Number of total products: {len(datasets)}")
# datasets = earthaccess.search_datasets(keyword=" ", cloud_hosted=False)
# print(f"Number of non-cloud hosted products: {len(datasets)}")
# datasets = earthaccess.search_datasets(keyword=" ", cloud_hosted=True)
# print(f"Number of cloud hosted products: {len(datasets)}")

# --- List the cloud_hosted dataproducts ---
# for dataset in datasets:
#     print(dataset["umm"]["ShortName"], dataset["umm"]["Version"])
    
# --- search and list all SPEXone data products ---
# spexone= earthaccess.search_datasets(instrument="spexone")
# for item in spexone:
#     summary = item.summary()
#     print(summary["short-name"])

# --- Search and filter for PACE aerosol data spatiotemporally ---
results = earthaccess.search_data(
    short_name="PACE_SPEXONE_L2_AER_RTAPLAND",
    version="3.0",
    cloud_hosted=True,
    bounding_box=(-23, 3, 24, 44), #WSEN
    temporal=("2024-06-01", "2024-06-01"),
    count=100  # Limit to files for simplicity
    # clouds = (0, 50) cloud cover if needs but for OCI
)
# --- number of filtered PACE aerosol files ---
print("Found", len(results), "files:")

# --- List the filtered filename and URL ---
for i, granule in enumerate(results, 1):
    file_name = granule["umm"]["GranuleUR"]
    data_links = granule.data_links(access="https")  # HTTPS for laptop
    # Use access="direct" in CryoCloud (us-west-2)
    file_url = data_links[0] if data_links else "No URL available"
    print(f"{i}. FileName: {file_name}")
    # print(f"   URL: {file_url}")
    
    
# --- Download filtered files ---
# local_path = ""
# local_path = os.getcwd()  # Set to current working directory
# os.makedirs(local_path, exist_ok=True)  # Create directory if it doesn't exist
# downloaded_files = earthaccess.download(results, local_path)
# print("Downloaded files:", [os.path.basename(f) for f in downloaded_files])


# # --- Stream all found files (with POCLOUD provider) ---
# files = earthaccess.open(results, provider="POCLOUD")
# print("Streamed files (with provider=POCLOUD):")#, [f.path for f in files])
# for f in files:
#     print(f.path)

# --- Stream without specifying provider (optional, for comparison) ---
files_auto = earthaccess.open(results)  # Auto-detects provider
print("Streamed files (auto-detected provider):")#, [f.path for f in files_auto])
for f in files_auto:
    print(f.path)
    
# --- Common Providers: --
# POCLOUD: Oceanography (e.g., PACE, SWOT).
# GESDISC: Atmospheric data (e.g., OMI, TROPOMI).
# NSIDCDAAC: Cryospheric data (e.g., ICESat-2).
# LPDAAC: Land data (e.g., MODIS, VIIRS).
# ASDC: Atmospheric science (e.g., CALIPSO).


# --- Data Aggregation for furthe analysis ---
all_lons = []
all_lats = []
all_aots_masked = []
print("\n--- Aggregating data from all files ---")
for f in files:
    print(f"Processing: {os.path.basename(f.path)}")
    try:
        with h5py.File(f, mode="r") as ds_h5py:
            if 'geophysical_data' in ds_h5py and 'aot550' in ds_h5py['geophysical_data']:
                aot550 = ds_h5py['geophysical_data']['aot550'][:]
                lat = ds_h5py['geolocation_data']['latitude'][:]
                lon = ds_h5py['geolocation_data']['longitude'][:]
                chi2 = ds_h5py['diagnostic_data']['chi2'][:]
                fill_value = ds_h5py['geophysical_data']['aot550'].attrs['_FillValue']

                # Add chi2 filter to the mask
                aot550_masked = np.ma.masked_where(
                    (aot550 == fill_value) | (aot550 < 0) | (aot550 > 5) | np.isnan(aot550) | (chi2 > 5),
                    aot550
                )
                if aot550_masked.count() > 0:
                    all_lons.append(lon)
                    all_lats.append(lat)
                    all_aots_masked.append(aot550_masked)
            else:
                print(f"--> Skipping: 'aot550' or 'geophysical_data' not found.")
    except Exception as e:
        print(f"--> Error reading file {os.path.basename(f.path)}: {e}")

# --- Plotting aod map---
if all_lons:
    print("\n--- Creating composite plot ---")
    fig = plt.figure(figsize=(12, 10))
    # Revert to PlateCarree to allow for a regional extent
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Dynamically set the extent to fit all plotted data
    min_lon = np.min([np.min(l) for l in all_lons])
    max_lon = np.max([np.max(l) for l in all_lons])
    min_lat = np.min([np.min(l) for l in all_lats])
    max_lat = np.max([np.max(l) for l in all_lats])
    ax.set_extent([min_lon - 1, max_lon + 1, min_lat - 1, max_lat + 1], crs=ccrs.PlateCarree())

    # Use stock image for a realistic background
    ax.stock_img()
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0, alpha=0.8)
    ax.add_feature(cfeature.OCEAN, zorder=0, alpha=0.8)
    # ax.add_feature(cfeature.COASTLINE, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.8)
    # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    mesh = None
    for lon, lat, aot in zip(all_lons, all_lats, all_aots_masked):
        # Use reversed 'hot' colormap
        mesh = ax.pcolormesh(
            lon, lat, aot, cmap='hot_r',
            vmin=0, vmax=1.5, transform=ccrs.PlateCarree(), zorder=2
        )

    plt.colorbar(mesh, label='Aerosol Optical Thickness (550 nm)', ax=ax, shrink=0.7)
    plt.title('PACE SPEXone AOT550 (2024-06-01) - Composite (chi2 <= 5)')

    output_path = "aot550_map_composite.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save with a standard white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Composite plot saved to: {output_path}")
    plt.close(fig)
else:
    print("\nNo valid data found in any file to plot.")