"""
Exporting a lit canopy as fspm-TF files: one frame per hour of a clear summer day, the format the other FSPM
platforms (CPlantBox, GroIMP, VirtualPlantLab) and ParaView read.

    python export_frames.py [--start H] [--hours N]      frames in ./outputs/frames: frame_<step>.glb, scene.fspm.json

The scene is the one openalea.caribu.simulation.simulate builds: a stand of 3 rows of 8 SimplePlants lit by
CaribuLightModel, hour by hour, under a clear day's PAR (21 June, Paris). Its recorder, given a frame_period, writes
the whole scene every step: every plant, organ and element with its variables, among them the light each element
absorbs (PARa), and the elements' triangles as a mesh. The sequence is then checked with fspm-TF's validator.

A second, filtered export shows logging: the plants and their elements only, with the light they absorb, in
./outputs/log/frames (outputs only: such frames are not reimported).

Open outputs/frames/scene.fspm.json in ParaView with fspm-TF's reader plugin (fspm-TF/paraview/fspm_tf_reader.py) and
colour by SubOrgan.PARa to watch the light move over the day; import_frames.py reads the files back.
"""
import argparse
import os

from openalea.fspm_tf.validate import validate_sequence
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.scene import Scene, SceneRecorder

from openalea.caribu.simulation import HOUR, CaribuLight, CaribuLightModel, SimplePlantModel, clear_day_PAR

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
ROWS, PLANTS_PER_ROW, ROW_SPACING, PLANT_SPACING = 3, 8, 0.15, 0.05
DAY_OF_YEAR = 172.
# a log of the light: two scales, the absorbed PAR and the lit area of each element
LOG = dict(scales=["Plant", "SubOrgan"], variables=["PARa", "lit_area", "leaf_rank"])


def export(start_hour: float = 6., hours: int = 12, outputs: str = OUTPUTS, frame_options: dict = None) -> str:
    """Light the stand hour by hour from *start_hour* (universal time) for *hours* hours, a frame per hour (filtered by
    *frame_options*, e.g. LOG); the sequence's index path."""
    plants = [Planting(x=PLANT_SPACING * (column + 0.5), y=ROW_SPACING * (row + 0.5), rotation=(37. * k) % 180.)
              for k, (row, column) in enumerate((row, column) for row in range(ROWS) for column in range(PLANTS_PER_ROW))]
    table = planting(plants, SimplePlantModel, {}, xrange=PLANTS_PER_ROW * PLANT_SPACING, yrange=ROWS * ROW_SPACING)
    clock_start = start_hour - 1.                      # a step lights the canopy at its end: start_hour first
    settings = {"start_day_of_year": DAY_OF_YEAR, "start_hour": clock_start, "lighting_interval": HOUR}
    incident = clear_day_PAR(DAY_OF_YEAR, CaribuLight.latitude)                   # PAR (µmol m-2 s-1) by hour
    built = Scene(table, environment=[CaribuLightModel], environment_scenarios=[settings], time_step=HOUR,
                  forcings={"PARi": lambda time: incident(clock_start + time / HOUR)})
    built.recorder = SceneRecorder(outputs, frame_period=1, frame_options=frame_options)   # outputs, with frames
    for _ in range(hours):
        built.run()
    return os.path.join(outputs, "frames", "scene.fspm.json")


def main():
    parser = argparse.ArgumentParser(description="Caribu's lit canopy as fspm-TF frames, one per hour")
    parser.add_argument("--start", type=float, default=6., help="the first hour (universal time)")
    parser.add_argument("--hours", type=int, default=12)
    arguments = parser.parse_args()
    index = export(arguments.start, arguments.hours)
    report = validate_sequence(index)
    frames = sorted(name for name in os.listdir(os.path.dirname(index)) if name.endswith(".glb"))
    print(f"{len(frames)} frames ({folder_size(os.path.dirname(index)):.1f} MB) and their index: {index}")
    print(f"fspm-TF validator: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    log = export(arguments.start, arguments.hours, os.path.join(OUTPUTS, "log"), frame_options=LOG)
    report = validate_sequence(log)
    print(f"logged (scales {LOG['scales']}, {len(LOG['variables'])} variables): {folder_size(os.path.dirname(log)):.1f} MB, "
          f"{len(report.errors)} error(s)")


def folder_size(folder: str) -> float:
    """The size of the .glb files in *folder* (MB)."""
    return sum(os.path.getsize(os.path.join(folder, name)) for name in os.listdir(folder) if name.endswith(".glb")) / 1e6


if __name__ == "__main__":
    main()
