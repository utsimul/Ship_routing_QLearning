import numpy as np

def compute_wind_speed_field(wind_u, wind_v):
    return np.sqrt(wind_u**2 + wind_v**2)