"""
Importing fspm-TF files: the frames export_frames.py wrote (run first when missing), read two ways.

    python import_frames.py [--show]          figure in ./outputs/absorbed_PAR.png

1. As any platform reads them, with openalea.fspm_tf alone: each element's absorbed PAR (PARa, in the unit its
   declaration gives) times its lit area gives the canopy's absorption per ground area, hour by hour, and its share
   by leaf rank.
2. Back into metafspm, with openalea.metafspm.io.fspm_tf.import_frame: the last frame rebuilds the stand's
   DataStructure (its MPG, its variables, bit for bit), whose absorbed light is the same.
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

from openalea import fspm_tf
from openalea.metafspm.io.fspm_tf import import_frame

from export_frames import OUTPUTS, PLANTS_PER_ROW, PLANT_SPACING, ROW_SPACING, ROWS, export

INDEX = os.path.join(OUTPUTS, "frames", "scene.fspm.json")
GROUND = PLANTS_PER_ROW * PLANT_SPACING * ROWS * ROW_SPACING                         # m2


def absorbed_by_hour(index: str = INDEX) -> pd.DataFrame:
    """The PAR absorbed per ground area (µmol m-2 s-1), in all and by leaf rank (0: the stems), one row per frame
    after the first, read with openalea.fspm_tf."""
    rows = []
    for frame in list(fspm_tf.read_sequence(index))[1:]:              # the first frame is before any lighting
        elements = frame.data_structures[0].entity_set("SubOrgan")
        declared = elements.variables["PARa"]
        assert declared.unit == "umol.m-2.s-1" and declared.standard_name == "absorbed_par_photon_flux_density", \
            (declared.unit, declared.standard_name)
        absorbed = np.nan_to_num(elements.columns["PARa"]) * elements.columns["lit_area"] / GROUND
        rank = elements.columns["leaf_rank"]
        row = {"hour (UTC)": 5. + frame.time / 3600., "canopy": float(absorbed.sum())}
        for value in np.unique(rank):
            row["stems" if value == 0 else f"leaf {int(value)}"] = float(absorbed[rank == value].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Caribu's fspm-TF frames read back")
    parser.add_argument("--show", action="store_true")
    show = parser.parse_args().show
    if not os.path.exists(INDEX):
        export()
    table = absorbed_by_hour()
    print("Read with openalea.fspm_tf (absorbed PAR per ground area, µmol m-2 s-1):")
    print(table.round(1).to_string(index=False))

    last = fspm_tf.read_index(INDEX)["frames"][-1]["path"]
    identifier, ds = next(iter(import_frame(last).items()))            # {id: DataStructure}, as metafspm wrote them
    rebuilt = float(np.sum(np.nan_to_num(ds.get("PARa")) * ds.get("lit_area")) / GROUND)
    print(f"\nRebuilt by metafspm from {os.path.basename(last)}: '{identifier}', a {type(ds).__name__} of "
          f"{ds.n_nodes()} elements; canopy absorption {rebuilt:.1f} µmol m-2 s-1")
    assert np.isclose(rebuilt, table["canopy"].iloc[-1], rtol=1e-12, atol=0)

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(6.5, 4))
    parts = [column for column in table if column not in ("hour (UTC)", "canopy")]
    table.plot.area(x="hour (UTC)", y=parts, ax=axes, stacked=True, alpha=0.8)
    axes.set_ylabel("absorbed PAR (µmol m⁻² s⁻¹ of ground)")
    axes.set_title("Caribu's canopy, read back from its fspm-TF frames")
    figure.tight_layout()
    path = os.path.join(OUTPUTS, "absorbed_PAR.png")
    figure.savefig(path, dpi=120)
    print(f"figure: {path}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
