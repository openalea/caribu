"""The near field's CUDA kernel under numba's CUDA simulator (in a subprocess: the simulator is chosen before numba
is imported) equals the CPU kernel: counts, far field and weights, full and nested radiosity, with the corrections'
option bits too."""
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent("""
    import numpy as np
    from openalea.caribu import formats, geometry, direct, nearfield, nested, sail
    from openalea.caribu.labels import is_translucent
    from openalea.caribu.comparison import port_run
    from openalea.caribu.legacy.data_samples import data_path
    vertices, labels = formats.read_can_arrays(data_path("f331s1_100plantes.can"))
    vertices, labels = vertices[:40], labels[:40]
    optics = port_run.optics({1: (0.10, 0.05)}, 0.15)
    points, labels, accepted, _ = direct.scene(vertices, labels)
    kept_points, kept = points[accepted], labels[accepted]
    translucent = is_translucent(kept)
    first_face, n_faces = direct.face_numbering(translucent)
    normals, centres = geometry.legacy_normals(kept_points), geometry.legacy_centres(kept_points)
    sensors = kept_points[:3].copy()
    sensors[:, :, 2] += 0.05
    sensor_normals, sensor_centres = geometry.legacy_normals(sensors), geometry.legacy_centres(sensors)
    flat = kept_points.reshape(-1, 3)
    pattern = (float(flat[:, 0].min()), float(flat[:, 1].min()), float(flat[:, 0].max()), float(flat[:, 1].max()))
    profile, _ = nested.far_field_profile(points, labels, [(1., (0., 0., -1.))], optics, pattern, 3,
                                          float(flat[:, 2].max()))
    optics_rows = np.tile([0.1, 0.05, 0.1, 0.05], (len(kept) + len(sensors), 1))
    for case, arguments in (("full", dict()),
                            ("nested", dict(grid=nearfield.grid(kept_points, sensors, 0.2, pattern), profile=profile,
                                            optics=optics_rows)),
                            ("nested -B", dict(grid=nearfield.grid(kept_points, sensors, 0.2, pattern),
                                               profile=profile, optics=optics_rows, bias=False)),
                            ("nested corrected", dict(grid=nearfield.grid(kept_points, sensors, 0.2, pattern),
                                                      profile=profile, optics=optics_rows,
                                                      options=nearfield.options(nearfield.OPTION_NAMES)))):
        radius = 2.0 if case == "full" else 0.2
        cpu = nearfield.counts(kept_points, kept, translucent, first_face, n_faces, normals, centres, radius,
                               sensors, sensor_normals, sensor_centres, size=12, **arguments)
        gpu = nearfield.counts(kept_points, kept, translucent, first_face, n_faces, normals, centres, radius,
                               sensors, sensor_normals, sensor_centres, size=12, device="cuda", **arguments)
        flatten = lambda value: [np.asarray(part) for part in (value if case == "full" else (*value[0], *value[1:]))]
        for a, b in zip(flatten(cpu), flatten(gpu)):
            assert a.shape == b.shape and np.array_equal(a, b), case
        print(case, "equal", len(flatten(cpu)[1]))
""")


def test_cuda_kernel_under_the_simulator():
    environment = dict(os.environ, NUMBA_ENABLE_CUDASIM="1", NUMBA_DISABLE_JIT="0")
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=environment, capture_output=True, text=True,
                          timeout=1800)
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.count("equal") == 4, done.stdout
