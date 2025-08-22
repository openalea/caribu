# -*- python -*-
#
#       Copyright 2015 INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       WebSite : https://github.com/openalea-incubator/caribu
#
# ==============================================================================
""" This module defines CaribuScene and CaribuSceneError classes."""

import os
import numpy
from itertools import groupby, chain
from math import sqrt
from numbers import Number

from openalea.mtg.mtg import MTG
from openalea.plantgl.all import Scene as pglScene, Viewer

from alinea.caribu.file_adaptor import read_can, read_light, read_pattern, \
    read_opt, build_materials
from alinea.caribu.plantgl_adaptor import scene_to_cscene, mtg_to_cscene
from alinea.caribu.caribu import raycasting, radiosity, mixed_radiosity, \
    x_raycasting, x_radiosity, x_mixed_radiosity, opt_string_and_labels, \
    triangles_string, pattern_string, write_scene
from alinea.caribu.display import jet_colors, generate_scene, nan_to_zero
from alinea.caribu.caribu_shell import vperiodise, Path
from functools import reduce
from alinea.caribu.caributriangleset import AbstractCaribuTriangleSet, CaribuTriangleSet 

import tempfile

def _agregate(values, indices, fun=sum):
    """ performs aggregation of outputs along indices """
    ag = {}
    for key, group in groupby(sorted(zip(indices, values), key=lambda x: x[0]),
                              lambda x: x[0]):
        vals = [elt[1] for elt in group]
        try:
            ag[key] = fun(vals)
        except TypeError:
            ag[key] = vals[0]
    return ag


def _convert(output, conv_unit):
    """ convert caribu output to meter/meter_square
    """
    # To DO ? filtering eabs_max / zero + occultations ?
    if conv_unit == 1:
        return output
    else:
        meter_square = conv_unit ** 2
        output['area'] = [area * meter_square for area in output['area']]
        for k in ['Eabs', 'Ei', 'Ei_sup', 'Ei_inf']:
            output[k] = [nrj / meter_square for nrj in output[k]]
        if 'sensors' in output:
            output['sensors']['area'] = [area * meter_square for area in output['sensors']['area']]
            for k in ['Ei', 'Ei0']:
                output['sensors'][k] = [nrj / meter_square for nrj in output['sensors'][k]]
        return output


def _wsum(nrj_area):
    nrj, area = list(zip(*nrj_area))
    area_tot = sum(area)
    if area_tot == 0:
        return 0
    else:
        return sum([e * a if a > 0 else 0 for e, a in nrj_area]) / area_tot


def domain_mesh(domain, z=0., subdiv=1):
    """ Create a triangle mesh covering a domain at height z

    Args:
        domain: (tuple of float) : a (xmin, ymin, xmax, ymax) tuple defining the extend of a square domain
        z: the altitude of the mesh
        subdiv: the number of subdivision of the mesh (not functional)

    Returns:
        a list of triangles. A triangle is a list of 3-tuples points coordinates
    """

    xmin, ymin, xmax, ymax = domain

    if subdiv > 1:
        raise UserWarning('subdivision of mesh not yet implemented')
        # TODO: copy code from cpp/canopy_io.cpp, function parse_can
    a = (xmin, ymin, z)
    b = (xmax, ymin, z)
    c = (xmin, ymax, z)
    d = (xmax, ymax, z)

    return [(a, b, c), (b, d, c)]


class CaribuScene:
    """A class interface to Caribu algorithms"""

    default_material = (0.06, 0.07)
    default_soil_reflectance = 0.15
    default_light = (1, (0, 0, -1))
    default_band = 'default_band'
    units = {'mm': 0.001, 'cm': 0.01, 'dm': 0.1, 'm': 1, 'dam': 10, 'hm': 100,
             'km': 1000}

    def __init__(self, scene=None, light=None, pattern=None, opt=None,
                 soil_reflectance=None, soil_mesh=None, z_soil=None,
                 scene_unit='m', debug = False, filecache = True):
        """ Initialise a CaribuScene

        Args:
            scene (dict): a {primitive_id: [triangles,]} dict.A triangle is a
                    list of 3-tuples points coordinates
                    Alternatively, scene can be a *.can file or a mtg with
                    'geometry' property or a plantGL scene.
                    For the later case, shape.id are used as primitive_id.
            light (list): a list of (Energy, (vx, vy, vz)) tuples defining light
                    sources
                    Alternatively,  a *.light file
                    If None (default), a unit energy vertical light is used.
                    Energy unit should be given per square-meter (m-2)
            pattern (tuple): 2D Coordinates of the domain bounding the scene for
                    its replication.
                     (xmin, ymin, xmax, ymax) scene is not bounded along z axis.
                     Alternatively a *.8 file.
                     if None (default), scene is not repeated
            opt (dict): a {band_name: {primitive_id: material}} dict of dict
                        or a {band_name: material} dict of tuples.
                        In the second form the material is used for all primitives.
                    A material is a 1-, 2- or 4-tuple depending on its optical behavior.
                    A 1-tuple encode an opaque material characterised by its reflectance
                    A 2-tuple encode a symmetric translucent material defined
                    by a reflectance and a transmittance
                    A 4-tuple encode an asymmetric translucent material defined
                    the reflectance and transmittance
                    of the upper and lower side respectively
                    Alternatively, a list of band_name.opt files (require scene
                    to be given as a *.can file)
                    If None (default), all primitive are associated to the
                    default material of the class.
            soil_reflectance (dict): a {band_name: reflectance} dict.
                    If None (default) class default soil reflectance is used for all bands
                    If *.opt files are provided, the values in opt files are used in priority
            soil_mesh (int): a flag triggering for the creation of a soil mesh
            in the scene during computations
                    If None (default) or -1, no soil is added
                    If an int (n), a soil is added to the scene, with n subdivisions
            z_soil (float): the altitude of the soil.
                    If None (default), the soil is placed at the bottom of
                    the scene bounding box
            scene_unit (str): the unit of length used for scene coordinate
            and for pattern (should be one of class.units default)
                    By default, scene_unit is considered to be 'm' (meter).

        Returns:
            A CaribuScene instance

        Note:
            File format specifications (*.can, *.light, *.8, *.opt) can be found in data/CanestraDoc.pdf
        """

        self.lightfile = None
        self.debug = debug

        if scene_unit not in self.units:
            raise ValueError('unrecognised scene unit: ' + scene_unit)
        self.conv_unit = self.units[scene_unit]

        self.scene = None
        if scene is not None:
            if isinstance(scene, dict):
                elt = scene[list(scene.keys())[0]]
                try:
                    assert isinstance(elt, list)
                    assert isinstance(elt[0], list)
                    assert isinstance(elt[0][0], tuple)
                except:
                    raise ValueError('Unrecognised scene format')
                self.scene = CaribuTriangleSet(scene)
            elif isinstance(scene, str):
                self.scene = CaribuTriangleSet(read_can(scene))
            elif isinstance(scene, MTG):
                self.scene = CaribuTriangleSet(mtg_to_cscene(scene))
            elif isinstance(scene, pglScene):
                self.scene = CaribuTriangleSet(scene_to_cscene(scene))
            elif isinstance(scene, AbstractCaribuTriangleSet):
                self.scene = scene
            else:
                raise ValueError('Unrecognised scene format')

        self.light = [self.default_light]
        if light is not None:
            self.setLight(light)

        self.pattern = None
        if pattern is not None:
            if isinstance(pattern, tuple):
                if len(pattern) == 2:
                    pattern = sum(pattern, ())
                if len(pattern) != 4:
                    raise ValueError('Unrecognised pattern format')
                self.pattern = pattern
            elif isinstance(pattern, str):
                self.pattern = read_pattern(pattern)
            else:
                raise ValueError('Unrecognised pattern format')

        self.material = None
        if opt is None:
            if soil_reflectance is None:
                self.soil_reflectance = {
                    self.default_band: self.default_soil_reflectance}
                bands = [self.default_band]
            else:
                self.soil_reflectance = soil_reflectance
                bands = list(soil_reflectance.keys())
            if self.scene is not None:
                self.material = {}
                for band in bands:
                    self.material[band] = {pid: self.default_material for pid in
                                           self.scene.keys()}
        else:
            if isinstance(opt, list):
                if not isinstance(opt[0], str):
                    raise ValueError('Unrecognised opt format')
                if not isinstance(scene, str):
                    raise ValueError(
                        'un-compatible inputs types: opt file and scene not a can file')
                if self.scene is not None:
                    self.material = {}
                    self.soil_reflectance = {}
                    for path in opt:
                        band = os.path.basename(path).split('.')[0]
                        n, ro_soil, po = read_opt(path)
                        self.material[band] = build_materials(list(self.scene.keys()),
                                                              po, ro_soil)
                        self.soil_reflectance[band] = ro_soil
            elif isinstance(opt, dict):
                elt = opt[list(opt.keys())[0]]
                if not isinstance(elt, dict):
                    if isinstance(elt, tuple):
                        self.material = {}
                        if self.scene is not None:
                            for band in opt:
                                self.material[band] = {pid: opt[band] for pid in
                                                       self.scene.keys()}
                    else:
                        raise ValueError('Unrecognised opt format')
                else:
                    self.material = opt
                if soil_reflectance is None:
                    self.soil_reflectance = {band: self.default_soil_reflectance
                                             for band in self.material}
                else:
                    if isinstance(soil_reflectance, dict):
                        if not len(soil_reflectance) == len(opt):
                            raise ValueError(
                                'the number of bands for optical properties and soil reflectance should match')
                        self.soil_reflectance = soil_reflectance
                    else:
                        raise ValueError('Unrecognised soil_reflectance format')
            else:
                raise ValueError('Unrecognised opt format')

        self.soil = None
        self.soil_label = 'soil'
        if self.scene is not None:
            ids = self.scene.allids()
            if ids is not None:
                if isinstance(ids[0], Number):
                    self.soil_label = -1
        if soil_mesh is not None:
            if soil_mesh != -1:
                if self.pattern is None:
                    raise ValueError(
                        'Adding a soil needs the scene domain to be defined')
                if z_soil is None:
                    if self.scene is None:
                        z_soil = 0
                    else:
                        z_soil = self.scene.getZmin()
                self.soil = domain_mesh(self.pattern, z_soil, soil_mesh)
        
        self.tempdir = '' # allow testing existence in __del__
        if filecache:
            self.tempdir = tempfile.mkdtemp() if not debug else './caribuscene_'+str(id(self))
        self.canfile = None
        self.optfile = None


    def __del__(self):
        if hasattr(self, 'tempdir') and self.tempdir:
            if os.path.exists(self.tempdir):
                import shutil
                shutil.rmtree(self.tempdir)



    def triangle_areas(self, convert=True):
        """ compute mean area of elementary triangles in the scene

        If convert is true, area is expressed in meter (scene unit otherwise)"""
        areas = self.scene.triangle_areas()
        if convert:
            areas *= self.conv_unit**2
        return areas

    def setLight(self, light):
        if isinstance(light, list):
            elt = light[0]
            try:
                assert isinstance(elt, tuple)
                assert isinstance(elt[1], tuple)
            except:
                raise ValueError('Unrecognised light format')
            self.light = light
            self.lightfile = None
        elif isinstance(light, str):
            self.light = read_light(light)
        else:
            raise ValueError('Unrecognised light format')

    def bbox(self):
        """ Scene bounding box opposite corner points

        Returns:
            two tuples: (xmin, ymin, zmin), (xmax, ymax, zmax)
        """
        return self.scene.getBoundingBox()

    def auto_screen(self, screen_resolution):
        pix = screen_resolution * self.conv_unit
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.bbox()
        ldiag = numpy.sqrt(
            (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
        return max(2, int(ldiag / pix))

    def plot(self, a_property=None, minval=None, maxval=None, gamma=None, display=True):
        """

        Args:
            a_property: {dict of float or dict of list of float} : a dict of values,
                each key being a scene primitive index.
            minval: (float) minimal value at lower bound of color range
                    if None (default), minimal value of property is used
            maxval: (float) maximal value at upper bound of color range
                    if None (default), maximal value of property is used
            gamma (float): exponent of the normalised values
                    if None (default), na gamma transform is applied
            display: (bool) : should the scene be displayed ? (default True)

        Returns:
            A plantGL scene
        """
        if a_property is None:
            color_property = None
            values = None
        else:
            values = list(a_property.values())
            if isinstance(values[0], list):
                values = list(chain.from_iterable(values))
            values = nan_to_zero(values)
            if minval is None:
                minval = min(values)
            if maxval is None:
                maxval = max(values)
            if gamma is None:
                gamma = 1
            norm = 1
            if minval != maxval:
                norm = maxval - minval
            values = [((x - minval) / float(norm))**gamma for x in values]
            colors = jet_colors(values, 0, 1)
            color_property = {}
            for k, v in a_property.items():
                if isinstance(v, list):
                    color_property[k] = []
                    for i in range(len(v)):
                        color_property[k].append(colors.pop(0))
                else:
                    color_property[k] = [colors.pop(0)] * self.scene.getNumberOfTriangles(k)
        scene = self.scene.generate_scene(color_property)
        if display:
            Viewer.display(scene)
        return scene, values



    def getIncidentEnergy(self):
        """ Compute energy of emission of light sources.

        Qi is the total horizontal irradiance emitted by sources (m-2)
        Qem is the sum of the normal-to-the-sources irradiance emitted by sources (m-2)
        Einc is the total incident energy received on the domain

        """

        Qi, Qem, Einc = None, None, None

        def _costheta(vect):
            vx, vy, vz = vect
            norme = sqrt(vx ** 2 + vy ** 2 + vz ** 2)
            return abs(vz / norme)

        if self.light is not None:
            nrj, direction = list(zip(*self.light))
            Qi = sum(nrj)
            costheta = list(map(_costheta, direction))
            Qem = sum([x[0] / x[1] for x in zip(nrj, costheta)])

            if self.pattern is not None:
                xmin, ymin, xmax, ymax = self.pattern
                d_area = abs((xmax - xmin) * (ymax - ymin)) * self.conv_unit**2
                Einc = Qi * d_area
        return Qi, Qem, Einc

    def getSoilEnergy(self):
        """ Compute energy received on soil.
        """
        Qi, Einc = None, None

        if self.soil is None:
            raise ValueError('A soil should be added to allow SoilEnargy Estimation')

        res = self.soil_aggregated
        if res is None:
            raise ValueError('Caribu should have been called  to allow SoilEnargy Estimation')

        # hack, TODO : take care of bands
        Qi = list(res.values())[0]['Ei']
        if self.pattern is not None:
            xmin, ymin, xmax, ymax = self.pattern
            d_area = abs((xmax - xmin) * (ymax - ymin)) * self.conv_unit**2
            Einc = Qi * d_area

        return Qi, Einc


    def as_primitive(self):
        """  Transform scene and materials into simpler python objects

        Returns:

        """
        triangles, groups, materials, bands, albedo = None, None, None, None, None
        if self.scene is not None:
            triangles = self.scene.allvalues(copied=True) #reduce(lambda x, y: x + y, self.scene.values())
            groups = self.scene.allids()
            if self.soil is not None:
                triangles += self.soil
                groups = groups + [self.soil_label] * len(self.soil)
            bands = list(self.material.keys())
            if len(bands) == 1:
                materials = self.scene.repeat_for_triangles([
                    self.material[bands[0]][pid] for
                    pid in self.scene.keys()])
                albedo = self.soil_reflectance[bands[0]]
                if self.soil is not None:
                    materials = materials + [(albedo,)] * len(self.soil)
            else:
                materials = {}
                for band in bands:
                    mat = self.scene.repeat_for_triangles([self.material[band][pid] for
                           pid in self.scene.keys()])
                    materials[band] = mat
                    if self.soil is not None:
                        materials = materials + [(self.soil_reflectance[
                                                      band],)] * len(self.soil)
                albedo = self.soil_reflectance

        return triangles, groups, materials, bands, albedo

    def run(self, direct=True, infinite=False, d_sphere=0.5, layers=10,
            height=None, screen_size=1536, screen_resolution=None, sensors=None,
            split_face=False, simplify=False):
        """ Compute illumination using the appropriate caribu algorithm

        Args:
            direct: (bool) Whether only first order interception is to be computed
                    Default is True (no rediffusions)
            infinite: (bool) Whether the scene should be considered as infinite
                    Default is False (non-infinite canopy)
            d_sphere: (float) the diameter (m) of the sphere defining the close
                     neighbourhood of mixed radiosity algorithm
                       if d_sphere = 0, direct + pure layer algorithm is used
            layers: (int) the number of horizontal layers for estimating far
                contributions
            height: (float) the height of the canopy (m).
                    if None (default), the maximal height of the scene is used.
            screen_size: (int) size of the screen_size x screen_size square
                    projection screen (pixels)
            screen_resolution: (float) real world size (meter) of a pixel of the
             projection screen. If None(default), screen_size is used.
            sensors: (dict of list of list of tuples) a {sensor_id: [triangle,...]} dict defining the virtual sensors
                each triangle is a list of tuple defining the coordinates of its vertices
            split_face: (bool) Whether results of incidence on individual faces
            of triangle should be output. Default is False
            simplify: (bool)  Whether results per band should be simplified to
            a {result_name: property} dict
                    in the case of a monochromatic simulation

        Returns:
            - raw (dict of dict) a {band_name: {result_name: property}} dict of dict.
            Except for result_name='sensors', each property is a {primitive_id: [values,]} dict containing results
             for individual triangles of the primitive
            - aggregated (dict of dict) : a {band_name: {result_name: property}}
            Except for result_name='sensors', each property is a {primitive_id: value} dict containing aggregated
             results for each primitive
            result_name are :
                      - area (float): the individual areas (m2)
                      - Eabs (float): the surfacic density of energy absorbed (m-2)
                      - Ei (float): the surfacic density of energy incoming  (m-2)
                      additionally, if split_face is True:
                      - Ei_inf (float): the surfacic density of energy incoming
                      on the inferior face (m-2)
                      - Ei_sup (float): the surfacic density of energy incoming
                       on the superior face (m-2)
                      - sensors (dict of dict): area, surfacic density of incoming
                       direct energy and surfacic density of incoming total energy
                       of sensors grouped by id, if any
        """

        raw, aggregated = {}, {}
        self.soil_raw, self.soil_aggregated = {}, {}
        results = ['Eabs', 'Ei', 'area']
        if split_face:
            results.extend(['Ei_inf', 'Ei_sup'])

        # convert lights to scene_unit
        lights = self.light
        if self.conv_unit != 1:
            lights = [(e * self.conv_unit ** 2, vect) for e, vect in self.light]

        if self.scene is not None:
            if self.debug:
                print ('Prepare scene', len(self.light))
            triangles = self.scene.allvalues(copied=True)
            groups = self.scene.allids()
            if self.debug:
                print ('done')
            if self.soil is not None:
                triangles += self.soil
                groups = groups + [self.soil_label] * len(self.soil)
            bands = list(self.material.keys())
            if len(bands) == 1:
                if not hasattr(self,'materialvalues') : 
                    materials = self.scene.repeat_for_triangles([
                       self.material[bands[0]][pid] for
                       pid in self.scene.keys()])
                    albedo = self.soil_reflectance[bands[0]]
                    if self.soil is not None:
                        materials = materials + [(albedo,)] * len(self.soil)
                    self.materialvalues = materials

                    if self.tempdir != '':
                        self.canfile = os.path.join(self.tempdir,'cscene.can')
                        self.optfile = os.path.join(self.tempdir,'band0.opt')
                        write_scene(triangles, materials, canfile = self.canfile, optfile = self.optfile)

                else:
                    # self.materialvalues is a cache for the computation of the material list
                    materials = self.materialvalues
                    albedo = self.soil_reflectance[bands[0]]

                algos = {'raycasting': raycasting, 'radiosity': radiosity,
                         'mixed_radiosity': mixed_radiosity}
            else:
                materials = {}
                if not hasattr(self,'materialvalues') : 
                    for band in bands:
                        mat = self.scene.repeat_for_triangles([self.material[band][pid] for
                               pid in self.scene.keys()])
                        materials[band] =  mat
                        if self.soil is not None:
                            materials = materials + [(self.soil_reflectance[
                                                          band],)] * len(self.soil)
                    albedo = self.soil_reflectance
                else:
                    materials = self.materialvalues
                    albedo = self.soil_reflectance

                algos = {'raycasting': x_raycasting, 'radiosity': x_radiosity,
                         'mixed_radiosity': x_mixed_radiosity}

            if not direct and infinite:  # mixed radiosity will be used
                if d_sphere < 0:
                    raise ValueError(
                        'calling radiosity should be done using direct=False and infinite=False')
                d_sphere /= self.conv_unit
                if height is None:
                    height = self.scene.getZmax()
                else:
                    height /= self.conv_unit

            if infinite and self.pattern is None:
                raise ValueError(
                    'infinite canopy illumination needs a pattern to be defined')

            if screen_resolution is not None:
                screen_size = self.auto_screen(screen_resolution)
                print('adjusted projection screen size: ' + str(screen_size))

            if sensors is not None:
                sensors_id = reduce(lambda x, y: x + y, [[k] * len(v) for k, v in sensors.items()], [])
                sensors = reduce(lambda x, y: x + y, list(sensors.values()), [])

            if not direct and infinite:  # mixed radiosity
                out = algos['mixed_radiosity'](triangles, materials,
                                               lights=lights,
                                               domain=self.pattern,
                                               soil_reflectance=albedo,
                                               diameter=d_sphere, layers=layers,
                                               height=height,
                                               screen_size=screen_size,
                                               sensors=sensors, debug = self.debug)
            elif not direct:  # pure radiosity
                out = algos['radiosity'](triangles, materials, lights=lights,
                                         screen_size=screen_size, sensors=sensors, debug = self.debug)
            else:  # ray_casting
                if infinite:
                    out = algos['raycasting'](triangles, materials,
                                              lights=lights,
                                              domain=self.pattern,
                                              screen_size=screen_size,
                                              sensors=sensors,
                                              debug = self.debug)
                else:
                    out = algos['raycasting'](triangles, materials,
                                              lights=lights, domain=None,
                                              screen_size=screen_size,
                                              sensors=sensors,
                                              debug = self.debug,
                                              canfile = self.canfile,
                                              optfile = self.optfile)

            if len(bands) == 1:
                out = {bands[0]: out}
            for band in bands:
                output = _convert(out[band], self.conv_unit)
                raw[band] = {}
                aggregated[band] = {}
                if 'sensors' in output:
                    sensors = output.pop('sensors')
                    raw[band]['sensors'] = {}
                    aggregated[band]['sensors'] = {}
                    for k in ('Ei', 'Ei0', 'area'):
                        raw[band]['sensors'][k] = _agregate(sensors[k], sensors_id, list)
                        if k == 'area':
                            aggregated[band]['sensors'][k] = _agregate(sensors[k], sensors_id, sum)
                        else:
                            aggregated[band]['sensors'][k] = _agregate(
                                zip(sensors[k], sensors['area']), sensors_id, _wsum)
                for k in results:
                    raw[band][k] = _agregate(output[k], groups, list)
                    if k == 'area':
                        aggregated[band][k] = _agregate(output[k], groups, sum)
                    else:
                        aggregated[band][k] = _agregate(
                            zip(output[k], output['area']), groups, _wsum)
                if self.soil is not None:
                    self.soil_raw[band] = {k: raw[band][k].pop(self.soil_label) for k in
                                           results}
                    self.soil_aggregated[band] = {
                        k: aggregated[band][k].pop(self.soil_label) for k in results}

            if simplify and len(bands) == 1:
                raw = raw[bands[0]]
                aggregated = aggregated[bands[0]]

        return raw, aggregated

    def runPeriodise(self):
        """ Call periodise and modify position of triangle in the scene to fit inside pattern"""
        triangles, groups, materials, bands, albedo = self.as_primitive()
        o_string, labels = opt_string_and_labels(materials)
        can_string = triangles_string(triangles, labels)
        pat_string = pattern_string(self.pattern)
        newscene = vperiodise(can_string, pat_string)
        infile = newscene.split('\n')
        cscene = {}
        for line in infile:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            fields = line.split()
            label = fields[2]
            if label not in cscene:
                cscene[label] = []
            coords = list(map(float, fields[-9:]))
            cscene[label].append(list(map(tuple, [coords[:3], coords[3:6], coords[6:]])))

        # should be changed to take into account the new structure
        self.scene = cscene

        return self
