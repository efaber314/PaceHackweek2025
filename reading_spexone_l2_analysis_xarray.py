import earthaccess
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive plotting for saving figures
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ssl
import os

# Bypass SSL verification for cartopy map data (safe for tutorials)
ssl._create_default_https_context = ssl._create_unverified_context

# Log in to NASA Earthdata
auth = earthaccess.login(strategy="netrc", persist=True)
print("Logged in:", auth.authenticated)

# Search for SPEXone Level-2 aerosol data for June 1, 2024
results = earthaccess.search_data(
    short_name="PACE_SPEXONE_L2_AER_RTAPLAND",
    version="3.0",
    cloud_hosted=True,
    bounding_box=(-23, 3, 24, 44),  # West, South, East, North
    temporal=("2024-06-01", "2024-06-01"),
    count=100
)
print("Found", len(results), "files")

# Download files to a local directory
download_dir = "./spexone_data"
os.makedirs(download_dir, exist_ok=True)
files = earthaccess.download(results, download_dir)
print("Downloaded files to:", download_dir)

# Lists to store aggregated data
all_lons = []
all_lats = []
all_aots = []

print("\n--- Aggregating data from all files ---")
for file_path in files:
    print(f"Processing: {os.path.basename(file_path)}")
    try:
        # Open data groups with xarray
        with xr.open_dataset(file_path, group='geophysical_data', decode_timedelta=True) as geophys_ds, \
             xr.open_dataset(file_path, group='geolocation_data', decode_timedelta=True) as geoloc_ds, \
             xr.open_dataset(file_path, group='diagnostic_data', decode_timedelta=True) as diag_ds:

            # Check for required variables
            if 'aot' not in geophys_ds or 'latitude' not in geoloc_ds or 'longitude' not in geoloc_ds or 'chi2' not in diag_ds:
                print(f"--> Skipping: Missing required variables")
                continue

            # Check for wavelength3d dimension
            if 'wavelength3d' not in geophys_ds['aot'].dims:
                print(f"--> Skipping: No wavelength3d dimension in aot")
                continue

            # Extract data for AOT at 550 nm (wavelength3d=7)
            aot = geophys_ds['aot'].isel(wavelength3d=7).values
            lat = geoloc_ds['latitude'].values
            lon = geoloc_ds['longitude'].values
            chi2 = diag_ds['chi2'].values

            # Ensure arrays have compatible shapes
            min_rows = min(aot.shape[0], chi2.shape[0], lat.shape[0], lon.shape[0])
            if min_rows == 0:
                print(f"--> Skipping: No valid data points")
                continue

            # Trim arrays to the same size
            aot = aot[:min_rows, :]
            chi2 = chi2[:min_rows, :]
            lat = lat[:min_rows, :]
            lon = lon[:min_rows, :]

            # Filter data: AOT between 0 and 5, chi2 <= 5, and valid numbers
            mask = (aot >= 0) & (aot <= 5) & (chi2 <= 5) & np.isfinite(aot)
            aot_valid = aot[mask]
            lat_valid = lat[mask]
            lon_valid = lon[mask]

            if aot_valid.size > 0:
                all_aots.extend(aot_valid)
                all_lats.extend(lat_valid)
                all_lons.extend(lon_valid)
                print(f"--> Found {aot_valid.size} valid data points")
            else:
                print(f"--> Skipping: No valid data after filtering")

    except Exception as e:
        print(f"--> Error reading file {os.path.basename(file_path)}: {e}")
        continue

# Create a composite plot
if all_lons:
    print("\n--- Creating composite plot ---")
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set map extent based on data
    min_lon, max_lon = np.min(all_lons), np.max(all_lons)
    min_lat, max_lat = np.min(all_lats), np.max(all_lats)
    ax.set_extent([min_lon - 1, max_lon + 1, min_lat - 1, max_lat + 1], crs=ccrs.PlateCarree())

    # Add map features
    ax.stock_img()
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.8)
    ax.add_feature(cfeature.OCEAN, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.8)

    # Plot AOT data as a scatter plot
    mesh = ax.scatter(
        all_lons, all_lats, c=all_aots,
        cmap='hot_r', vmin=0, vmax=1.5, transform=ccrs.PlateCarree(), s=1
    )

    plt.colorbar(mesh, label='Aerosol Optical Thickness (550 nm)', shrink=0.7)
    plt.title('PACE SPEXone AOT550 (2024-06-01) - Composite')

    # Save the plot
    output_path = "/Users/piyushkmrp/Documents/1_work/2_meetings/PACE_Hackweek/Pre-event_tutorial/earthdata_access/aot550_map_composite_xarray.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Composite plot saved to: {output_path}")
    plt.close(fig)
else:
    print("\nNo valid data found to plot.")