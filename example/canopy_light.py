"""
CaribuLight in a metafspm scene: a stand of 20 wheat plants grown by Adel's port (ShootArchitecture, its elements'
triangles), lit by CaribuLightModel every 4 hours under the sky and the sun of clear days, from 6 May at Paris'
latitude; in between, each element's absorbed PAR follows the incident PAR with its last Erel (Wheat-BRIDGES' use).
Each scene step is one hour and 10 °Cd of growth.

Drawn (canopy_light.png in the working directory): the canopy coloured by Erel at the last lighting, and the stand's
absorbed PAR per m² of ground hour by hour against the incident PAR.

    python example/canopy_light.py [n_steps]          (needs openalea.adel: pip install -e ../adel --no-deps)
"""
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from openalea.adel.development import ScenarioShoot, random_delays, sown_stand
from openalea.caribu import lights
from openalea.caribu.component import CaribuLightModel
from openalea.metafspm.display import draw, value_colors
from openalea.metafspm.scene.population import planting
from openalea.metafspm.scene.scene import Scene

N_PLANTS, N_ROWS, ROW_SPACING, PLANT_SPACING = 20, 4, 0.15, 0.04
XRANGE, YRANGE = N_PLANTS // N_ROWS * PLANT_SPACING, N_ROWS * ROW_SPACING
DAY_OF_YEAR, LATITUDE = 126, 48.85
DAILY_GLOBAL_RADIATION = 28e6            # J m-2 day-1, a clear May day at Paris
PAR_PER_GLOBAL, MICROMOLES_PER_JOULE = 0.48, 4.57


def incident_PAR(time):
    """The incident PAR (µmol m-2 s-1) at the scene's *time* (s from midnight of DAY_OF_YEAR)."""
    hour = (time / 3600.) % 24
    day = DAY_OF_YEAR + time // 86400
    global_radiation = max(lights.hourly_global_radiation(DAILY_GLOBAL_RADIATION, hour, day, LATITUDE), 0.)
    return PAR_PER_GLOBAL * MICROMOLES_PER_JOULE * global_radiation


def stand_scene():
    plants = sown_stand(N_PLANTS, n_rows=N_ROWS, row_spacing=ROW_SPACING, plant_spacing=PLANT_SPACING)
    table = planting(plants, ScenarioShoot, {"n_sect": 2, "thermal_time_step": 10.},
                     [{"delay": float(delay)} for delay in random_delays(N_PLANTS)], xrange=XRANGE, yrange=YRANGE)
    light = {"sun": True, "start_day_of_year": DAY_OF_YEAR, "latitude": LATITUDE, "lighting_interval": 4 * 3600.}
    return Scene(table, environment=[CaribuLightModel], environment_scenarios=[light], time_step=3600.,
                 forcings={"PARi": incident_PAR})


if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    scene = stand_scene()
    ds = scene.populations[0].data_structure
    light = scene.environment[0].light
    hours, incident, absorbed = [], [], []
    started = time.perf_counter()
    for step in range(n_steps):
        scene.run()
        area = np.asarray(ds.get("lit_area"))
        hours.append(scene.time / 3600.)
        incident.append(light.forcing("PARi"))
        absorbed.append(float(np.sum(np.asarray(ds.get("PARa")) * area)) / (XRANGE * YRANGE))
    elapsed = time.perf_counter() - started
    triangles = ds.triangles()

    figure = plt.figure(figsize=(13, 5))
    axes = figure.add_subplot(1, 2, 1, projection="3d")
    erel = np.asarray(ds.get("Erel"))
    draw(axes, triangles, colors=value_colors(erel, limits=(0., 1.)), shade=False)
    figure.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=axes, shrink=0.6, label="Erel")
    axes.set_title(f"Erel, the absorbed PAR relative to the incident ({n_steps} h, {n_steps * 10} °Cd)")
    axes = figure.add_subplot(1, 2, 2)
    axes.plot(hours, incident, label="incident PAR")
    axes.plot(hours, absorbed, label="absorbed by the stand, per m² of ground")
    axes.set_xlabel("hour")
    axes.set_ylabel("µmol m-2 s-1")
    axes.legend()
    figure.savefig("canopy_light.png", dpi=120)
    lit = np.asarray(ds.get("lit_area")) > 0
    print(f"canopy_light.png: {N_PLANTS} plants, {len(triangles)} triangles at the end; {n_steps} hours in "
          f"{elapsed:.1f} s; the stand absorbs {sum(absorbed) / max(sum(incident), 1e-12):.2f} of the incident PAR; "
          f"Erel of the lit elements from {erel[lit].min():.2f} to {erel[lit].max():.2f}")
