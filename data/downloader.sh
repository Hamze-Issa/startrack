# Shell script to download data from the ESA Marine Service
# for Sea surface temperature, Sea surface salinity, and Phytoplankton
# for the area and time from the Fresh4Bio project Activity 1 (east greenland, 2007-2024)
# https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/download
# https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_SSS_L3_MYNRT_015_014/download


# Sea surface temperature
copernicusmarine subset \
--dataset-id METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2 \
--variable analysed_sst \
--variable analysis_error \
--variable mask \
--variable sea_ice_fraction \
--start-datetime 2024-01-01T00:00:00 \
--end-datetime 2024-05-01T00:00:00 \
--minimum-longitude -40 \
--maximum-longitude 0 \
--minimum-latitude 65 \
--maximum-latitude 80

# Sea surface salinity
copernicusmarine subset \
--dataset-id cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D \
--variable Sea_Surface_Salinity \
--variable Mean_Acq_Time \
--start-datetime 2024-01-01T00:00:00 \
--end-datetime 2024-05-01T00:00:00 \
--minimum-longitude -40 \
--maximum-longitude 0 \
--minimum-latitude 65 \
--maximum-latitude 80
