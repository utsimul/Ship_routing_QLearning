import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',

        'variable': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'mean_sea_level_pressure',
            'total_precipitation'
        ],

        'year': '2023',

        'month': '01',

        'day': [
            '01','02','03','04','05'
        ],

        'time': [
            '00:00','06:00','12:00','18:00'
        ],

        'format': 'netcdf',
    },

    'weather.nc')