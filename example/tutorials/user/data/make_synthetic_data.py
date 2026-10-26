"""
The tutorials' example data: SYNTHETIC, not measurements. They stand for what an experiment would give, so that the
user tutorial 05 can show how to give the model a measured input, compare it with observations and fit a parameter.

- weather_PAR.csv: the incident PAR (µmol m-2 s-1) hour by hour (universal time) of a partly cloudy 21 June at
  Paris: a clear day's PAR (openalea.caribu.simulation.clear_day_PAR) times a random cloudiness between 0.5 and 1;
- observed_soil_PAR.csv: the PAR reaching the soil under the stand, from 8 to 16 h, as a line of light sensors laid
  on the ground would measure it: made by simulate() with the sun, that weather and leaves 2.2 cm wide (the default is
  1.5 cm), with 5 % noise.

    python make_synthetic_data.py          (rewrites the two files in this folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from openalea.caribu.simulation import clear_day_PAR, simulate

HERE = Path(__file__).parent
TRUE_LEAF_WIDTH = 0.022


def main():
    noise = np.random.default_rng(0)
    hours = np.arange(24)
    clear = clear_day_PAR(day_of_year=172., latitude=48.85)
    weather = pd.DataFrame({"hour": hours, "PARi": np.round(
        [clear(hour) * noise.uniform(0.5, 1.) for hour in hours], 1)})
    weather.to_csv(HERE / "weather_PAR.csv", index=False)

    PARi = weather.set_index("hour")["PARi"]
    result = simulate({"sun": True, "leaf_width": TRUE_LEAF_WIDTH}, hours=9, start_hour=8, PARi=PARi)
    observed = result.summary[["hour"]].copy()
    observed["soil_PAR"] = np.round(result.summary["PARi"] * result.summary["soil_fraction"]
                                    * noise.normal(1., 0.05, len(observed)), 1)
    observed.to_csv(HERE / "observed_soil_PAR.csv", index=False)


if __name__ == "__main__":
    main()
