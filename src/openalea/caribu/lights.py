"""
Light sources as caribu takes them, [(E, (vx, vy, vz))]: E the irradiance of a horizontal plane, (vx, vy, vz) the
beam's direction. The sky as the turtle's 16 or 46 directions or as sectors of azimuth × zenith, each weighted by
the standard (soc) or uniform (uoc) overcast sky's luminance; the sun's position from the day, hour and latitude;
the diffuse fraction of hourly global radiation (Spitters et al. 1986); hourly global radiation from the daily one
(Kaplanis 2006). The legacy's light.py and sky_tools, their conventions kept (their quirks in doc/legacy.rst,
Q-L01 to Q-L06; the correction "sky" fixes them):

    lights = turtle_lights("46", "soc", energy=1.)                     the turtle sky
    sky = Sky(4, 5); sky.diffuse(1., "soc"); lights = sky.lights()      GenSky, GetLight
    lights = sun_light(1., day_of_year, hour_utc, latitude)             Gensun, GetLightsSun
    fraction = diffuse_fraction(global_radiation, day_of_year, hour_utc, latitude)   spitters_horaire.RdRsH

Two azimuth conventions coexist, as in the legacy: light.py's directions are the azimuth from X+ counter-clockwise
(`light_sources` from North, clockwise, with an orientation of X+); the sector sky's and the sun's are radians,
the sun's azimuth from North, the sky's sector angles from X+.

With the correction "sky" (*corrections*, corrections.resolve's): one convention, `light_sources`' (azimuths from
North, clockwise; *orientation* the angle, clockwise, from X+ to North), for the turtle, the sector sky and the sun
(its true solar azimuth); the turtle's beam energies E_h / sin(elevation); the sun with π, none at night (zenith angle
beyond 90°), placed in its own sector; merge_skies keeping the first sky; normalised keeping an added sun.
"""
import math

import numpy as np

from . import corrections as corrections_module


def _corrected(corrections) -> bool:
    return "sky" in corrections_module.resolve(corrections)


def _north_direction(elevation, azimuth, orientation):
    """The beam's direction from a source at *elevation* (degrees) and *azimuth* (degrees from North, clockwise),
    X+ at *orientation* degrees clockwise from North (light_sources')."""
    return direction(elevation, -(np.asarray(azimuth) + orientation))

# --- The turtle (light.py) ---

ELEVATIONS_46 = [9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 10.81, 10.81, 10.81, 10.81, 10.81, 26.57,
                 26.57, 26.57, 26.57, 26.57, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08,
                 47.41, 47.41, 47.41, 47.41, 47.41, 52.62, 52.62, 52.62, 52.62, 52.62, 69.16, 69.16, 69.16, 69.16,
                 69.16, 90]
AZIMUTHS_46 = [12.23, 59.77, 84.23, 131.77, 156.23, 203.77, 228.23, 275.77, 300.23, 347.77, 36, 108, 180, 252, 324, 0,
               72, 144, 216, 288, 23.27, 48.73, 95.27, 120.73, 167.27, 192.73, 239.27, 264.73, 311.27, 336.73, 0, 72,
               144, 216, 288, 36, 108, 180, 252, 324, 0, 72, 144, 216, 288, 180]
WEIGHTS_46 = {"soc": [0.0043] * 10 + [0.0055] * 5 + [0.014] * 5 + [0.0197] * 10 + [0.0336] * 5 + [0.0399] * 5
                     + [0.0495] * 5 + [0.0481],
              "uoc": [0.007] * 10 + [0.0086] * 5 + [0.017] * 5 + [0.0224] * 10 + [0.0317] * 5 + [0.036] * 5
                     + [0.0405] * 5 + [0.0377]}
# the 46 directions' solid-angle weights (sunDome's)
SOLID_ANGLE_WEIGHTS_46 = ([0.026808309] * 10 + [0.029325083] * 5 + [0.031299545] * 5 + [0.038160959] * 10
                          + [0.045638829] * 5 + [0.050212264] * 5 + [0.052965108] * 5 + [0.0481])
ELEVATIONS_16 = [90, 26.57, 26.57, 26.57, 26.57, 26.57, 52.62, 52.62, 52.62, 52.62, 52.62, 10.81, 10.81, 10.81,
                 10.81, 10.81]
AZIMUTHS_16 = [180, 0, 72, 144, 216, 288, 36, 108, 180, 252, 324, 36, 108, 180, 252, 324]
WEIGHTS_16 = {"soc": [0.1468] + [0.0448] * 5 + [0.108] * 5 + [0.01777] * 5,
              "uoc": [0.1173] + [0.0533] * 5 + [0.0981] * 5 + [0.0251] * 5}


def _turtle_table(sectors="46", sky_type="soc", energy=1.):
    """(elevation, azimuth, horizontal energy) of each direction of a turtle: "46", "16", or one zenith source."""
    if sectors == "46":
        elevations, azimuths, weights = ELEVATIONS_46, AZIMUTHS_46, WEIGHTS_46["soc" if sky_type == "soc" else "uoc"]
    elif sectors == "16":
        elevations, azimuths, weights = ELEVATIONS_16, AZIMUTHS_16, WEIGHTS_16["soc" if sky_type == "soc" else "uoc"]
    else:
        elevations, azimuths, weights = [90], [0], [1]
    return [(elevation, azimuth, weight * energy) for elevation, azimuth, weight in
            zip(elevations, azimuths, weights)]


def direction(elevation, azimuth):
    """vecteur_direction: the beam's direction from a source at *elevation* (degrees above the horizon) and
    *azimuth* (degrees from X+, counter-clockwise), arrays or numbers."""
    theta = np.radians(90 - np.array(elevation))
    phi = np.radians(azimuth)
    return -np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi), -np.cos(theta)


def emission_inv(elevation, energy):
    """The irradiance of a horizontal plane under a source of *energy* (normal to the beam) at *elevation*."""
    return energy * abs(np.cos(np.radians(90 - elevation)))


def turtle(sectors="46", sky_type="soc", energy=1., corrections=None, orientation=0.):
    """light.turtle: (horizontal energies, energies along the beam (emission_inv's, its arguments swapped as the
    legacy's call), directions, elevations, azimuths) of a turtle sky of *energy* on the horizontal. Correction
    "sky": the beam energies E_h / sin(elevation), the azimuths from North (light_sources')."""
    corrected = _corrected(corrections)
    rows = []
    for elevation, azimuth, weight in _turtle_table(sectors, sky_type, energy):
        if corrected:
            rows.append((weight, weight / math.sin(math.radians(elevation)),
                         _north_direction(elevation, azimuth, orientation), elevation, azimuth))
        else:
            rows.append((weight, emission_inv(weight, elevation), direction(elevation, azimuth), elevation, azimuth))
    return list(zip(*rows))


def turtle_lights(sectors="46", sky_type="soc", energy=1., corrections=None, orientation=0.) -> list:
    """The turtle sky's light sources [(E, (vx, vy, vz))]."""
    energies, _, directions, _, _ = turtle(sectors, sky_type, energy, corrections, orientation)
    return list(zip(energies, directions))


def diffuse_source(directions=1) -> list:
    """light.diffuse_source: a turtle sky of unit energy (1, 16 or 46 directions)."""
    return turtle_lights(str(directions), "soc", 1)


def light_sources(elevation, azimuth, irradiance, orientation=0) -> list:
    """light.light_sources: sources at *elevation* (degrees), *azimuth* (degrees from North, clockwise) of horizontal
    *irradiance*; *orientation* the angle (degrees, clockwise) from X+ to North."""
    if not isinstance(elevation, (list, tuple, np.ndarray)):
        elevation = [elevation]
    if not isinstance(azimuth, (list, tuple, np.ndarray)):
        azimuth = [azimuth]
    if not isinstance(irradiance, (list, tuple, np.ndarray)):
        irradiance = [irradiance]
    x, y, z = direction(elevation, -(np.array(azimuth) + orientation))
    return [(energy, (xx, yy, zz)) for energy, xx, yy, zz in zip(np.array(irradiance), np.array(x), np.array(y),
                                                                  np.array(z))]


def light_source(horizontal_irradiance, elevation, azimuth) -> list:
    """light.light_source: one source (*azimuth* from X+, counter-clockwise), as a list (the legacy returned it in
    a tuple, its trailing comma)."""
    return [(horizontal_irradiance, direction(elevation, azimuth))]


# --- The sector sky (sky_tools: Sky, GenSky, GetLight, place_sun, add_sun, merge, Normalise_Sky) ---

def _sector_direction(zenith, azimuth):
    """Sky.Vdir: the beam's direction from a sector at *zenith* and *azimuth* (radians, from X+)."""
    return [math.sin(zenith) * math.cos(azimuth), math.sin(zenith) * math.sin(azimuth), -math.cos(zenith)]


def uniform_overcast(zenith, zenith_width, azimuth, azimuth_width):
    """Sky.uoc: the share of a uniform overcast sky's horizontal irradiance in a sector (radians)."""
    zenith_width /= 2.
    x, y = math.cos(zenith - zenith_width), math.cos(zenith + zenith_width)
    return (x * x - y * y) * azimuth_width / 2. / math.pi


def standard_overcast(zenith, zenith_width, azimuth, azimuth_width):
    """Sky.soc: the share of a standard overcast sky's (Moon & Spencer) horizontal irradiance in a sector."""
    zenith_width /= 2.
    x, y = math.cos(zenith - zenith_width), math.cos(zenith + zenith_width)
    return (3 / 14. * (x * x - y * y) + 6 / 21. * (x * x * x - y * y * y)) * azimuth_width / math.pi


class Sky:
    """Sky: *azimuths* × *zeniths* sectors (azimuth outer, from X+; zenith inner, from the zenith), each a source
    of energy 0 until set; rows [E, vx, vy, vz, azimuth index, zenith index]. Correction "sky": the sectors'
    azimuths from North, clockwise (X+ at *orientation* degrees from North), the sun placed in its own sector."""

    def __init__(self, azimuths, zeniths, corrections=None, orientation=0.):
        self.sectors = [azimuths, zeniths]
        self.corrections, self.orientation = corrections, orientation
        self.corrected = _corrected(corrections)
        self.azimuth_width = 2 * math.pi / azimuths
        self.zenith_width = math.pi / 2 / zeniths
        self.rows = []
        for j in range(azimuths):
            for k in range(zeniths):
                azimuth = j * self.azimuth_width + self.azimuth_width / 2
                zenith = k * self.zenith_width + self.zenith_width / 2
                self.rows.append([0] + self._direction(zenith, azimuth) + [j, k])

    def _direction(self, zenith, azimuth):
        """The beam from a sector or the sun at *zenith* and *azimuth* (radians)."""
        if not self.corrected:
            return _sector_direction(zenith, azimuth)
        return [float(value) for value in _north_direction(90 - math.degrees(zenith), math.degrees(azimuth),
                                                           self.orientation)]

    def diffuse(self, energy, sky_type="soc"):
        """set_Rd: the sectors' energies, *energy* on the horizontal shared as the soc (or uoc) sky."""
        share = standard_overcast if sky_type == "soc" else uniform_overcast
        count = 0
        for j in range(self.sectors[0]):
            for k in range(self.sectors[1]):
                azimuth = j * self.azimuth_width + self.azimuth_width / 2
                zenith = k * self.zenith_width + self.zenith_width / 2
                self.rows[count][0] = share(zenith, self.zenith_width, azimuth, self.azimuth_width) * energy
                count += 1
        return self

    def place_sun(self, sun):
        """set_Rsun: the sun's energy added to the sector it is in (Which_SkySec's, its quirks kept: the zenith
        index compared to the sectors' azimuth index and the other way round; night never detected)."""
        if self.corrected:
            zenith, azimuth = sun.position()
            if not 0 <= zenith < math.pi / 2:
                return self                              # the night
            j = int((azimuth % (2 * math.pi)) // self.azimuth_width) % self.sectors[0]
            k = min(int(zenith // self.zenith_width), self.sectors[1] - 1)
            self.rows[j * self.sectors[1] + k][0] += sun.energy
            return self
        sector = sector_of(self.sectors[0], self.sectors[1], sun.position())
        for row in self.rows:
            if row[4] == sector[0] and row[5] == sector[1]:
                row[0] += sun.energy
        return self

    def add_sun(self, sun):
        """set_Rsun2: the sun added as a source of its own, at its exact position."""
        zenith, azimuth = sun.position()
        if self.corrected and not 0 <= zenith < math.pi / 2:
            return self
        self.rows.append([sun.energy] + self._direction(zenith, azimuth) + [-1, -1])
        return self

    def __add__(self, other):
        """Two skies' sectors added (None when their sectors differ, as the legacy)."""
        if self.sectors != other.sectors:
            return None
        total = Sky(*self.sectors, self.corrections, self.orientation)
        for index in range(len(self.rows)):
            total.rows[index][0] = self.rows[index][0] + other.rows[index][0]
        return total

    def total(self):
        """get_Rs: the sky's total energy."""
        result = 0
        for row in self.rows:
            result = result + row[0]
        return result

    def normalised(self):
        """Noralised_Sky: a sky of the same sectors, its energies summing to 1 (all 0 when the sky is dark; the
        sun added by add_sun dropped, as the legacy)."""
        total = self.total()
        result = Sky(*self.sectors, self.corrections, self.orientation)
        if self.corrected:                                  # an added sun kept
            result.rows = [list(row) for row in self.rows]
        if total > 0.:
            for index in range(len(result.rows)):
                result.rows[index][0] = self.rows[index][0] / total
        return result

    def lights(self) -> list:
        """GetLight: the sources [(E, (vx, vy, vz))] (its text read back: the same floats)."""
        return [(float(row[0]), (float(row[1]), float(row[2]), float(row[3]))) for row in self.rows]


def merge_skies(skies):
    """mergeNskies: the skies' sum, from an empty sky of the first's sectors and the others (the first's energies
    left out, as the legacy; the first's corrections: "sky" keeps them)."""
    first = skies[0]
    total = Sky(*first.sectors, first.corrections, first.orientation)
    for sky in (skies if first.corrected else skies[1:]):
        total += sky
    return total


def sector_of(azimuths, zeniths, position):
    """Which_SkySec: (zenith index, azimuth index) of the sector boundaries nearest to the sun's *position*
    (zenith, azimuth radians, the azimuth from North in [−π, π]), [−1, −1] for a negative zenith."""
    zenith, azimuth = position
    if zenith < 0:
        return [-1, -1]
    zenith_bounds = np.array(list(range(zeniths))) * (math.pi / (2 * zeniths))
    azimuth_bounds = np.array(list(range(azimuths))) * (2 * math.pi / azimuths) - math.pi
    zenith_distances = abs(zenith_bounds - zenith).tolist()
    azimuth_distances = abs(azimuth_bounds - azimuth).tolist()
    return [zenith_distances.index(min(zenith_distances)), azimuth_distances.index(min(azimuth_distances))]


# --- The sun (sky_tools: Sun, Gensun, GetLightsSun) ---

class Sun:
    """Sun: its *energy* (horizontal irradiance), its zenith angle and azimuth (radians; the legacy calls the
    zenith angle `elev`). Correction "sky": π, the azimuth the solar one (from North, clockwise), none at night, its
    direction in light_sources' frame (X+ at *orientation* degrees, clockwise, from North)."""

    def __init__(self, energy=1.0, zenith=0., azimuth=0., corrections=None, orientation=0.):
        self.energy, self.zenith, self.azimuth = energy, zenith, azimuth
        self.corrected, self.orientation = _corrected(corrections), orientation

    def place(self, day_of_year, hour_utc, latitude):
        """_set_pos_astro: the position at *hour_utc* (solar time) of *day_of_year* at *latitude* (radians), with
        the legacy's 3.14 for π in its declination and zenith."""
        declination = sun_declination(day_of_year, self.corrected)
        hour_angle = 2 * math.pi / 24 * (hour_utc - 12)
        half_pi = math.pi / 2 if self.corrected else 3.14 / 2
        self.zenith = half_pi - math.asin(math.sin(latitude) * math.sin(declination)
                                          + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle))
        first = math.sin(latitude) * math.cos(hour_angle) - math.cos(latitude) * math.tan(declination)
        if self.corrected:  # from North, clockwise: atan2 gives it from South, westwards
            self.azimuth = (math.pi + math.atan2(math.sin(hour_angle), first)) % (2 * math.pi)
        else:
            self.azimuth = -math.atan2(math.sin(hour_angle), first)
        return self

    def position(self):
        return self.zenith, self.azimuth

    def lights(self) -> list:
        """toLight: the sun as a source [(E, (vx, vy, vz))], none when its zenith angle is not positive (the
        legacy's test: a sun below the horizon still shines, upwards)."""
        if self.corrected:
            if not 0 <= self.zenith < math.pi / 2:
                return []
            vector = _north_direction(90 - math.degrees(self.zenith), math.degrees(self.azimuth), self.orientation)
            return [(float(self.energy), tuple(float(value) for value in vector))]
        if not self.zenith > 0:
            return []
        return [(float(self.energy), (math.sin(self.zenith) * math.cos(self.azimuth),
                                      math.sin(self.zenith) * math.sin(self.azimuth), -math.cos(self.zenith)))]


def sun_declination(day_of_year, corrected=False):
    """Sun.DecliSun: the sun's declination (radians), with the legacy's 3.14 (π when *corrected*)."""
    alpha = 2 * (math.pi if corrected else 3.14) * (day_of_year - 1) / 365
    return 0.006918 - 0.399912 * math.cos(alpha) + 0.070257 * math.sin(alpha)


def sun_light(energy, day_of_year, hour_utc, latitude, corrections=None, orientation=0.) -> list:
    """Gensun and GetLightsSun: the sun of *energy* at *hour_utc* of *day_of_year* at *latitude* (degrees, turned
    into radians with 3.14 as the legacy; π with the correction "sky")."""
    to_radians = (math.pi if _corrected(corrections) else 3.14) / 180
    return Sun(energy, corrections=corrections, orientation=orientation).place(day_of_year, hour_utc,
                                                                               to_radians * latitude).lights()


# --- Radiation (spitters_horaire) ---

def declination(day_of_year):
    """DecliSun of spitters_horaire: the sun's declination (radians), with π."""
    alpha = 2 * math.pi * (day_of_year - 1) / 365
    return 0.006918 - 0.399912 * math.cos(alpha) + 0.070257 * math.sin(alpha)


def day_length(latitude, sun_declination_value):
    """DayLength: the day's length as an angle (radians) at *latitude* (degrees)."""
    angle = math.acos(-math.tan(sun_declination_value) * math.tan(math.radians(latitude)))
    if angle < 0:
        angle = angle + math.pi
    return 2 * angle


def angle_hours(angle):
    """dH: a day length angle (radians) in hours."""
    return 24 / (2 * math.pi) * angle


def extraterrestrial(day_of_year, hour_utc, latitude):
    """extra: the extraterrestrial irradiance of a horizontal plane (W m-2) at *hour_utc* (solar)."""
    hour_angle = 2 * math.pi / 24 * (hour_utc - 12)
    lat = math.radians(latitude)
    sun_declination_value = declination(day_of_year)
    cosine = (math.sin(lat) * math.sin(sun_declination_value)
              + math.cos(lat) * math.cos(sun_declination_value) * math.cos(hour_angle))
    solar_constant = 1370 * (1 + 0.033 * math.cos(2 * math.pi * (day_of_year - 4) / 366))
    return solar_constant * cosine


def diffuse_fraction(global_radiation, day_of_year, hour_utc, latitude):
    """RdRsH: the hourly diffuse fraction of global radiation (Spitters et al. 1986), from its ratio to the
    extraterrestrial irradiance."""
    hour_angle = 2 * math.pi / 24 * (hour_utc - 12)
    lat = math.radians(latitude)
    sun_declination_value = declination(day_of_year)
    cosine = (math.sin(lat) * math.sin(sun_declination_value)
              + math.cos(lat) * math.cos(sun_declination_value) * math.cos(hour_angle))
    solar_constant = 1370 * (1 + 0.033 * math.cos(2 * math.pi * (day_of_year - 4) / 366))
    ratio = global_radiation / (solar_constant * cosine)
    clear = 0.847 - 1.61 * cosine + 1.04 * cosine * cosine
    limit = (1.47 - clear) / 1.66
    if ratio <= 0.22:
        return 1
    if ratio <= 0.35:
        return 1 - 6.4 * (ratio - 0.22) ** 2
    if ratio <= limit:
        return 1.47 - 1.66 * ratio
    return clear


def hourly_global_radiation(daily_global_radiation, hour_utc, day_of_year, latitude):
    """RgH (Kaplanis 2006): the global irradiance (W m-2: the legacy's docstring says J m-2 h-1, its formula's
    Rg / 86400 gives an irradiance) at *hour_utc* of a day of *daily_global_radiation* (J m-2 day-1), between
    sunrise and sunset. The legacy's could not run (`(a / b) ^ 2` on floats); its π of 3.14116 kept."""
    sun_declination_value = declination(day_of_year)
    lat = math.radians(latitude)
    pi = 3.14116
    a = math.sin(lat) * math.sin(sun_declination_value)
    b = math.cos(lat) * math.cos(sun_declination_value)
    psi = pi * daily_global_radiation / 86400 / (a * math.acos(-a / b) + b * math.sqrt(1 - (a / b) ** 2))
    return -b * psi * math.cos(2 * pi * hour_utc / 24) + a * psi
