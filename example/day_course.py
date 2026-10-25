"""
A day of light on a wheat canopy (the 100-plant Adel scene, infinite), hour by hour: the global radiation of a clear
day spread over the hours (Kaplanis), its diffuse fraction (Spitters), the diffuse part on a standard overcast sky
of 4 × 5 sectors, the direct part from the sun's position. The canopy is prepared once (a Canopy): the sky's
directions are projected at the first hour and reused, so each later hour costs only its sun's projection.

    python example/day_course.py
"""
import time

import numpy as np

from openalea.caribu import formats, lights
from openalea.caribu.canopy import Canopy
from openalea.caribu.legacy.data_samples import data_path

DAY_OF_YEAR, LATITUDE = 140, 48.85          # 20 May, Paris
DAILY_GLOBAL_RADIATION = 22e6               # J m-2 day-1
PAR_FRACTION = 0.48                         # PAR in the global radiation
OPTICS = (0.15, {1: (0.10, 0.10, 0.05, 0.10, 0.05)})   # soil ρ; the species' (ρ stem, ρ, τ upper, ρ, τ lower)

vertices, labels = formats.read_can_arrays(data_path("f331s1_100plantes.can"))
flat = vertices.reshape(-1, 3)
pattern = (float(flat[:, 0].min()), float(flat[:, 1].min()), float(flat[:, 0].max()), float(flat[:, 1].max()))
canopy = Canopy(vertices, labels, pattern=pattern)
plants = (labels // 10 ** 6) % 10 ** 5

absorbed = np.zeros(len(labels))           # J m-2 of each triangle over the day (PAR)
incident = 0.
started = time.perf_counter()
for hour in np.arange(5.5, 19.0, 1.0):
    sun = lights.Sun().place(DAY_OF_YEAR, hour, np.radians(LATITUDE))
    if sun.zenith >= np.pi / 2:
        continue                            # below the horizon
    global_radiation = max(lights.hourly_global_radiation(DAILY_GLOBAL_RADIATION, hour, DAY_OF_YEAR, LATITUDE), 0)
    par = PAR_FRACTION * global_radiation  # W m-2 on the horizontal
    diffuse = lights.diffuse_fraction(global_radiation, DAY_OF_YEAR, hour, LATITUDE)
    sky = lights.Sky(4, 5).diffuse(par * diffuse, "soc").lights()
    sources = sky + lights.sun_light(par * (1 - diffuse), DAY_OF_YEAR, hour, LATITUDE)
    result = canopy.direct(sources, OPTICS)
    absorbed += np.nan_to_num(result["Eabs"].astype(np.float64)) * 3600
    incident += par * 3600
    print(f"{hour:5.1f} h  PAR {par:6.1f} W m-2  diffuse {diffuse:4.2f}  canopy absorbs "
          f"{np.nansum(result['Eabs'] * result['area']) / np.nansum(result['area']):6.1f} W m-2 of leaf")
print(f"the day in {time.perf_counter() - started:.1f} s")

area = canopy.direct(sources, OPTICS)["area"]
per_plant = np.bincount(plants, weights=absorbed * area) / np.bincount(plants, weights=area).clip(1e-12)
present = np.bincount(plants) > 0
print(f"incident PAR {incident / 1e6:.2f} MJ m-2; absorbed per m² of leaf, plants' mean "
      f"{per_plant[present].mean() / 1e6:.2f}, min {per_plant[present].min() / 1e6:.2f}, "
      f"max {per_plant[present].max() / 1e6:.2f} MJ m-2")
