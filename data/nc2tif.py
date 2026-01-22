import numpy as np
import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import einops as eo
from datetime import datetime, timedelta
import re
import netCDF4 as nc
import os
from datetime import datetime, timedelta
import shutil
import argparse

def read_nc(file_path, name_list):
    # Read the NetCDF file
    nc_dataset = Dataset(file_path, 'r')
    vars = {name: np.array(nc_dataset.variables[f'{name}'][:], dtype=np.float64) for name in name_list}

    # if necessary, replace the fill value with nan (the data has lots of no_data that are filled with 'fill_value' which could be any number
    # so we replace them with nan to make them clearly distinct and not mistake them for real data)
    for var in vars.keys():
        if len(vars[var].shape) < 2: continue
        print(f'Replacing the fill value {nc_dataset.variables[var][0].fill_value} with nan for variable {var}')
        var_fill_value = nc_dataset.variables[var][0].fill_value
        # vars[var][vars[var] == var_fill_value] = np.nan
        # vars[var] = np.where(vars[var][:].data == var_fill_value, np.nan, vars[var])
        vars[var] = np.where(vars[var][:] > 1e36, np.nan, vars[var])

    # Close the NetCDF dataset
    nc_dataset.close()
    return vars

def check_disk_space(path, min_gb=30):
    try:
        usage = shutil.disk_usage(path)
        if usage.free < min_gb * 1024**3:
            raise RuntimeError(f"Insufficient space in {path} (less than {min_gb}GB free)")
    except FileNotFoundError:
        raise RuntimeError(f"Path {path} not accessible")

def read_nc_light(file_path, name_list):
    # Read the NetCDF file
    nc_dataset = Dataset(file_path, 'r')
    vars = {name: np.array(nc_dataset.variables[f'{name}'][:], dtype=np.float64) for name in name_list}
    # Close the NetCDF dataset
    nc_dataset.close()
    return vars

def read_nc_light_range(file_path, name_list, start, end):
    # Read the NetCDF file
    nc_dataset = Dataset(file_path, 'r')
    vars = {name: np.array(nc_dataset.variables[f'{name}'][start:end, :], dtype=np.float64) for name in name_list}
    for var in vars.keys():
        if len(vars[var].shape) < 2: continue
        print(f'Replacing negative values with nan for variable {var}')
        var_fill_value = nc_dataset.variables[var][0].fill_value
        # vars[var][vars[var] == var_fill_value] = np.nan
        # vars[var] = np.where(vars[var][:].data == var_fill_value, np.nan, vars[var])
        vars[var] = np.where(vars[var][:] < 0, np.nan, vars[var])
    # Close the NetCDF dataset
    nc_dataset.close()
    return vars

def get_all_files(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)
    return file_list

def get_time_range(filename):
    match = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        year, month, day = match.groups()
        try:
            # Parse the extracted date
            date = datetime.strptime(f"{year}{month}{day}", "%Y%m%d")
            # Get the start and end of the month
            start_of_month = date.replace(day=1)
            next_month = (start_of_month + timedelta(days=31)).replace(day=1)
            # Format as filename_yyyymmddT000000_yyyymmddT000000
            date_range = f"{start_of_month.strftime('%Y%m%dT000000')}_{next_month.strftime('%Y%m%dT000000')}"
        except ValueError:
            print(f'No date found in file {filename}')

    return date_range

def save_vars_to_tiff(vars, filename, output_folder):
    x_resolution = np.abs(vars['x'][1] - vars['x'][0])
    y_resolution = np.abs(vars['y'][1] - vars['y'][0])
    transform = from_origin(vars['x'].min(), vars['y'].max(), x_resolution, y_resolution)  # top-left x, top-left y, x resolution, y resolution
    crs = 'EPSG:3031' # says so in the nc file
    time = get_time_range(filename)
    for var in vars.keys():
        if len(vars[var].shape) < 2: continue # this only works for variables with at least 2 dimensions (x, y)
        var_folder = f'{var}/'
        if not os.path.exists(output_folder + var_folder):
            os.makedirs(output_folder + var_folder)
        geotiff_file = f'{var}_{time}.tif'
        with rasterio.open(
            output_folder + var_folder + geotiff_file, 'w', driver='GTiff', height=vars[var].shape[0],
            width=vars[var].shape[1], count=1, dtype='float64',
            crs=crs, transform=transform
        ) as dst:
            dst.write(vars[var], 1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variable",
        type=str,
        required=True,
        help="NetCDF variable name, e.g. 'analysed_sst' or 'Sea_Surface_Salinity'",
    )
    parser.add_argument(
        "--nc-file",
        type=str,
        required=True,
        help="Path to NetCDF file to process",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="./tiff",
        help="Root output folder (default: ./tiff)",
    )
    return parser.parse_args()


# variable = "analysed_sst" # ["analysed_sst", 'Sea_Surface_Salinity']
# variable_folder = variable # ['sst', 'sss']
# # nc_file = 'cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D_multi-vars_39.80W-0.00W_65.00N-79.80N_2024-01-01-2025-01-01.nc'
# nc_file = f'METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2_multi-vars_39.97W-0.03W_65.03N-79.97N_2024-01-17-2025-01-01.nc'


def main():
    args = parse_args()

    # Map variable to folder label if needed
    # e.g. analysed_sst -> sst, Sea_Surface_Salinity -> sss, CHL -> chl
    variable = args.variable
    nc_file = args.nc_file
    variable_folder = variable

    chl_min, chl_max = 0, 65.0
    sss_min, sss_max = -2.71, 62.61
    sst_min, sst_max = 270.145, 287.8

    output_folder = f"./tiff/{variable_folder}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the NetCDF file
    with nc.Dataset(nc_file, "r") as dataset:
        # Extract variables
        var = dataset.variables[variable]  # variable (time x lat x lon)
        time_var = dataset.variables["time"]  # Time variable (days since 1900-01-01)
        lat = dataset.variables["latitude"][:]  # Latitude array
        lon = dataset.variables["longitude"][:]  # Longitude array

        # Get the number of time slices
        num_time_slices = var.shape[0]

        # Define chunk size (e.g., 100 time slices at a time)
        chunk_size = 100

        # Process in chunks
        for start in range(0, num_time_slices, chunk_size):
            end = min(start + chunk_size, num_time_slices)
            print(f"Processing time slices {start} to {end - 1}...")

            # Read a chunk of the variable
            var_chunk = var[start:end, :, :]
            if variable == "CHL":
                var_chunk = np.where(
                    np.isfinite(var_chunk),  # Check if the value is finite
                    np.log1p(var_chunk),  # Use log1p to handle zeros: log(1 + x)
                    var_chunk  # Keep NaN values unchanged
                )

            # Normalize only finite values, keep NaN values as they are
            if variable_folder == 'CHL': var_min, var_max = np.log1p(chl_min), np.log1p(chl_max)
            if variable_folder == 'analysed_sst': var_min, var_max = sst_min, sst_max
            if variable_folder == 'Sea_Surface_Salinity': var_min, var_max = sss_min, sss_max

            var_chunk = np.where(
                np.isfinite(var_chunk),  # Check if the value is finite
                (var_chunk - var_min) / (var_max - var_min),  # Normalize finite values
                var_chunk  # Keep NaN values unchanged
            )
            print(f"Normalized for variable {variable_folder}, and min max values {var_min}, {var_max}")

            # Read the corresponding time values
            time_chunk = time_var[start:end]

            # Convert time values to datetime objects
            time_units = time_var.units
            time_calendar = time_var.calendar if hasattr(time_var, "calendar") else "standard"
            times = nc.num2date(time_chunk, units=time_units, calendar=time_calendar)

            # Save each time slice as a GeoTIFF
            for i in range(var_chunk.shape[0]):
                # Get the timestamp in ddmmyyyy format
                timestamp = times[i]

                if timestamp.year > 2009:
                    # Create the output GeoTIFF filename
                    timestamp_str = timestamp.strftime("%d%m%Y")
                    output_file = os.path.join(output_folder, f"{variable_folder}_{timestamp_str}.tif")

                    # Define spatial metadata
                    transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))

                    # check_disk_space('/mnt/experiment-3/')

                    # Save the time slice as a GeoTIFF
                    with rasterio.open(
                        output_file,
                        "w",
                        driver="GTiff",
                        height=var_chunk.shape[1],
                        width=var_chunk.shape[2],
                        count=1,
                        dtype=var_chunk.dtype,
                        crs="EPSG:4326",  # Assuming WGS84 coordinate system
                        transform=transform,
                    ) as dst:
                        dst.write(np.flipud(var_chunk[i, :, :]), 1)
                else:
                    print(f"Skipping {timestamp} (before 2010 or after 2018)")

            print(f"Saved time slices {start} to {end - 1}.")

    print("Processing complete.")


    folder_path = f"./tiff/{variable_folder}"

    # Function to convert ddmmyyyy to yyyymmddT000000_yyyymmddT000000
    def convert_filename(filename):
        # Extract the date part from the filename (assuming it's in the format sst_ddmmyyyy.tif)
        stem, _ = os.path.splitext(filename)          # "Sea_Surface_Salinity_01022024"
        # Take the part after the LAST underscore
        date_part = stem.rsplit("_", 1)[-1]           # "01022024"
        
        # Parse the date in ddmmyyyy format
        date_obj = datetime.strptime(date_part, "%d%m%Y")
        
        # Format the first date as yyyymmddT000000
        first_date = date_obj.strftime("%Y%m%dT000000")
        
        # Calculate the next day and format it as yyyymmddT000000
        next_date = date_obj + timedelta(days=1)
        second_date = next_date.strftime("%Y%m%dT000000")
        
        # Construct the new filename
        new_filename = f"{variable_folder}_{first_date}_{second_date}.tif"
        
        return new_filename

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            # Get the full path of the file
            old_file_path = os.path.join(folder_path, filename)
            
            # Generate the new filename
            new_filename = convert_filename(filename)
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            # print(f"Renamed: {filename} -> {new_filename}")

    print("Renaming complete.")


if __name__ == "__main__":
    main()
