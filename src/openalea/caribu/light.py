"""
Utilities to create lights
"""
from numpy import radians, sin, cos, array, ndarray

elevations = [9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 9.23, 10.81, 10.81, 10.81, 10.81, 10.81, 26.57,
              26.57, 26.57, 26.57, 26.57, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08, 31.08, 47.41,
              47.41, 47.41, 47.41, 47.41, 52.62, 52.62, 52.62, 52.62, 52.62, 69.16, 69.16, 69.16, 69.16, 69.16, 90]

azimuths = [12.23, 59.77, 84.23, 131.77, 156.23, 203.77, 228.23, 275.77, 300.23, 347.77, 36, 108, 180, 252, 324, 0, 72,
            144, 216, 288, 23.27, 48.73, 95.27, 120.73, 167.27, 192.73, 239.27, 264.73, 311.27, 336.73, 0, 72, 144, 216,
            288, 36, 108, 180, 252, 324, 0, 72, 144, 216, 288, 180]

# weights = [0.026808309, 0.026808309, 0.026808309, 0.026808309, 0.026808309, 0.026808309, 0.026808309, 0.026808309,
#            0.026808309, 0.026808309, 0.029325083, 0.029325083, 0.029325083, 0.029325083, 0.029325083, 0.031299545,
#            0.031299545, 0.031299545, 0.031299545, 0.031299545, 0.038160959, 0.038160959, 0.038160959, 0.038160959,
#            0.038160959, 0.038160959, 0.038160959, 0.038160959, 0.038160959, 0.038160959, 0.045638829, 0.045638829,
#            0.045638829, 0.045638829, 0.045638829, 0.050212264, 0.050212264, 0.050212264, 0.050212264, 0.050212264,
#            0.052965108, 0.052965108, 0.052965108, 0.052965108, 0.052965108, 0.0481]

weights_soc = [0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0055, 0.0055, 0.0055,
               0.0055, 0.0055, 0.014, 0.014, 0.014, 0.014, 0.014, 0.0197, 0.0197, 0.0197, 0.0197, 0.0197, 0.0197,
               0.0197, 0.0197, 0.0197, 0.0197, 0.0336, 0.0336, 0.0336, 0.0336, 0.0336, 0.0399, 0.0399, 0.0399, 0.0399,
               0.0399, 0.0495, 0.0495, 0.0495, 0.0495, 0.0495, 0.0481]
weights_uoc = [0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.0086, 0.0086, 0.0086, 0.0086,
               0.0086, 0.017, 0.017, 0.017, 0.017, 0.017, 0.0224, 0.0224, 0.0224, 0.0224, 0.0224, 0.0224, 0.0224,
               0.0224, 0.0224, 0.0224, 0.0317, 0.0317, 0.0317, 0.0317, 0.0317, 0.036, 0.036, 0.036, 0.036, 0.036,
               0.0405, 0.0405, 0.0405, 0.0405, 0.0405, 0.0377]

elevations16 = [90, 26.57, 26.57, 26.57, 26.57, 26.57, 52.62, 52.62, 52.62, 52.62, 52.62, 10.81, 10.81, 10.81, 10.81,
                10.81]
azimuths16 = [180, 0, 72, 144, 216, 288, 36, 108, 180, 252, 324, 36, 108, 180, 252, 324]
weights16_soc = [0.1468, 0.0448, 0.0448, 0.0448, 0.0448, 0.0448, 0.108, 0.108, 0.108, 0.108, 0.108, 0.01777, 0.01777,
                 0.01777, 0.01777, 0.01777]
weights16_uoc = [0.1173, 0.0533, 0.0533, 0.0533, 0.0533, 0.0533, 0.0981, 0.0981, 0.0981, 0.0981, 0.0981, 0.0251, 0.0251,
                 0.0251, 0.0251, 0.0251]


def _turtle(sectors='46', format='soc', energy=1.):
    """ return a generator on parameters of a given turtle """
    if sectors == '46':
        el = elevations
        az = azimuths
        if format == 'soc':
            w = weights_soc
        else:
            w = weights_uoc
    elif sectors == '16':
        el = elevations16
        az = azimuths16
        if format == 'soc':
            w = weights16_soc
        else:
            w = weights16_uoc
    else:
        el = [90]
        az = [0]
        w = [1]
    for i in range(len(el)):
        yield el[i], az[i], w[i] * energy


def turtle(sectors='46', format='soc', energy=1.):
    """ Return energy on horizontal surface, energy in the direction of emission,
    direction vector, elevation (deg) and azimuth (deg) for a turtle discretion
    of the sky hemisphere"""
    res = [(energy, emission_inv(energy, elevation), (vecteur_direction(elevation, azimuth)), elevation, azimuth) for
           elevation, azimuth, energy in _turtle(sectors, format, energy)]
    return list(zip(*res))


def vecteur_direction(elevation, azimuth):
    """ coordinate of look_at source vector from elevation and azimuth (deg, f
    rom X+ positive counter-clockwise)"""
    theta = radians(90 - array(elevation))
    phi = radians(azimuth)
    return -sin(theta) * cos(phi), -sin(theta) * sin(phi), -cos(theta)


def light_sources(elevation, azimuth, irradiance, orientation=0):
    """Create caribu light sources

    Args:
        elevation: (array-like) elevation angle (degree, positive) from
         horizontal of sources
        azimuth: (array-like) : azimuth angle (degree, from North, positive
         clockwards) of sources
        irradiance: (array-like): horizontal irradiance of sources
        orientation: (float)  the angle (deg, positive clockwise) from X+ to
         North (default: 0)

    Returns:
        a list of (irradiance, (x, y, z)) tuples
    """
    if not isinstance(elevation, (list, tuple, ndarray)):
        elevation = [elevation]
    if not isinstance(azimuth, (list, tuple, ndarray)):
        azimuth = [azimuth]
    if not isinstance(irradiance, (list, tuple, ndarray)):
        irradiance = [irradiance]
    az = -(array(azimuth) + orientation)
    x, y, z = vecteur_direction(elevation, az)
    return [(irr, (xx, yy, zz)) for irr, xx, yy, zz in
            zip(array(irradiance), array(x), array(y), array(z))]


def light_source(horizontal_irradiance, elevation, azimuth):
    """ return a  punctual infinite light source for caribu

    Args:
        horizontal_irradiance: horizontal irradiance of the source
        elevation: elevation angle (deg)
        azimuth: azimuth angle (deg). positive counter-clockwise from X+

    Returns:
        a list of one light source ready to use for caribu
    """

    return [(horizontal_irradiance, vecteur_direction(elevation, azimuth))],


def emission_inv(elevation, energy):
    """ return energy of emission for a source of a given direction and of a given energy
    received on a horizontal surface """
    theta = radians(90 - elevation)
    received_energy = energy * abs(cos(theta))
    return received_energy


def diffuse_source(directions=1):
    energie, emission, direction, elevation, azimuth = turtle(sectors=str(directions), energy=1)
    return list(zip(energie, direction))
