import xarray as xr
import numpy as np

data = xr.open_dataset("data_stream-oper_stepType-instant.nc")

u = data["u10"].values
v = data["v10"].values

wind = np.sqrt(u**2 + v**2)

# downsample
wind_small = wind[::7, ::7]

np.save("weather_base.npy", wind_small)