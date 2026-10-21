import numpy as np
from pykrige.ok import OrdinaryKriging
import folium
from folium import plugins
import matplotlib.pyplot as plt

def generate_kriging_heatmap(df_wards, pollutant_col="Current_AQI", grid_res=35):
    """
    Fits an Ordinary Kriging model on ward coordinates and returns
    interpolated grid coordinates and weights for Folium HeatMap.
    """
    lons = df_wards["lon"].values
    lats = df_wards["lat"].values
    vals = df_wards[pollutant_col].values

    # Fit Ordinary Kriging
    ok = OrdinaryKriging(
        lons, lats, vals,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )

    grid_lon = np.linspace(lons.min() - 0.03, lons.max() + 0.03, grid_res)
    grid_lat = np.linspace(lats.min() - 0.03, lats.max() + 0.03, grid_res)

    z, ss = ok.execute('grid', grid_lon, grid_lat)

    # Format into Folium HeatMap points: [[lat, lon, weight], ...]
    heat_data = []
    max_val = np.nanmax(z)
    for i, lat in enumerate(grid_lat):
        for j, lon in enumerate(grid_lon):
            val = float(z[i, j])
            if not np.isnan(val) and val > 0:
                # Normalized intensity weight (0.0 to 1.0)
                norm_weight = min(max(val / max_val, 0.1), 1.0)
                heat_data.append([lat, lon, norm_weight])

    return heat_data
