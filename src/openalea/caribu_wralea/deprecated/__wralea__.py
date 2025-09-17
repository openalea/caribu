# This file has been generated at Sat Aug 04 18:13:50 2012

from openalea.core import Factory as Fa
from openalea.core import (IBool, IData, IDict, IFileStr, IFloat, IInt,
                           ISequence, IStr)

__name__ = 'openalea.caribu.deprecated'

__editable__ = True
__description__ = ' Caribu package '
__license__ = ''
__url__ = ''
__version__ = '0.0.4'
__authors__ = 'M. Chelle,C. Fournier'
__institutes__ = 'INRA'

__all__ = []

GenOutput = Fa(uid="5cc625664e8411e6bff6d4bed973e64a",
               name='GenOutput',
               authors='M. Chelle,C. Fournier (wralea authors)',
               description='deprecated node',
               nodemodule='openalea.caribu.deprecated_nodes',
               nodeclass='GenOutput', inputs=[
        {'interface': IStr, 'name': 'Etri', 'value': None, 'desc': ''},
        {'interface': IStr, 'name': 'Eabs', 'value': None, 'desc': ''}],
               outputs=[
                   {'interface': None, 'name': 'OutDict',
                    'desc': ''}], widgetmodule=None,
               widgetclass=None)
__all__.append('GenOutput')

FileCaribuScene = Fa(uid="5cc625674e8411e6bff6d4bed973e64a",
                     name='FileCaribuScene',
                     authors='C. Fournier',
                     description='deprecated node',
                     nodemodule='openalea.caribu.deprecated_nodes',
                     nodeclass='FileCaribuScene',
                     inputs=[{'interface': IFileStr(
                         filter="*.can", save=False),
                         'name': 'CanFile',
                         'value': None,
                         'desc': ''}, {
                         'interface': IFileStr(
                             filter="*.light",
                             save=False),
                         'name': 'LightFile',
                         'value': None,
                         'desc': ''}, {
                         'interface': IFileStr(
                             filter="*.8",
                             save=False),
                         'name': 'PatternFile',
                         'value': None,
                         'desc': ''}, {
                         'interface': IFileStr(
                             filter="*.opt",
                             save=False),
                         'name': 'OptFile',
                         'value': None,
                         'desc': ''}],
                     outputs=[{'interface': IStr,
                               'name': 'CaribuScene',
                               'desc': ''},
                              {'interface': IDict,
                               'name': 'Maping dict (see CaribuScene)'}],
                     widgetmodule=None,
                     widgetclass=None, )
__all__.append('FileCaribuScene')

ObjCaribuScene = Fa(uid="5cc625684e8411e6bff6d4bed973e64a",
                    name='ObjCaribuScene',
                    authors='M. Chelle,C. Fournier (wralea authors)',
                    description='deprecated node',
                    nodemodule='openalea.caribu.deprecated_nodes',
                    nodeclass='ObjCaribuScene',
                    inputs=[{'interface': IData,
                             'name': 'CanObj',
                             'value': None,
                             'desc': 'Canopy object with to_canestra method'},
                            {'interface': IStr,
                             'name': 'LightString',
                             'value': None,
                             'desc': ''},
                            {'interface': ISequence,
                             'name': 'pattern',
                             'value': None,
                             'desc': 'Tuple describig pattern'},
                            {'interface': IStr,
                             'name': 'optstring',
                             'value': None,
                             'desc': 'Optical properties'},
                            {'interface': IStr,
                             'name': 'wavelength',
                             'value': None,
                             'desc': 'Current wavelength'}],
                    outputs=[{'interface': IStr,
                              'name': 'CaribuScene',
                              'desc': ''},
                             {'interface': IDict,
                              'name': 'Mapid',
                              'desc': ''}],
                    widgetmodule=None,
                    widgetclass=None, )
__all__.append('ObjCaribuScene')

getEi = Fa(uid="5cc625694e8411e6bff6d4bed973e64a",
           name='get Ei',
           authors='M. Chelle,C. Fournier (wralea authors)',
           description='deprecated node',
           nodemodule='openalea.caribu.deprecated_nodes',
           nodeclass='getEi',
           inputs=[{'interface': None, 'name': 'CaribuScene'}],
           outputs=[{'interface': IFloat, 'name': 'Qi',
                     'desc': ('incident light flux received on an horizontal '
                              'surface (per scene unit area)')}],
           widgetmodule=None, widgetclass=None, )
__all__.append('getEi')

MCSail_MCSail = Fa(uid="5cc6256a4e8411e6bff6d4bed973e64a",
                   name='MCSail',
                   authors='M. Chelle,C. Fournier (wralea authors)',
                   description='Compute mean fluxes on layered canopy',
                   category='deprecated',
                   nodemodule='openalea.caribu.deprecated_nodes',
                   nodeclass='MCSail',
                   inputs=[{'interface': None, 'name': 'Sailscene'},
                           {'interface': IBool, 'name': 'Sleep',
                            'value': False}],
                   outputs=[{'interface': IStr, 'name': 'Fluxes'},
                            {'interface': IStr, 'name': 'Log'}],
                   widgetmodule=None,
                   widgetclass=None,
                   )
__all__.append('MCSail_MCSail')

S2v_S2v = Fa(uid="5cc6256b4e8411e6bff6d4bed973e64a",
             name='S2v',
             authors='M. Chelle,C. Fournier (wralea authors)',
             description='Transform a 3D CaribuScene into a 1D Sail Scene',
             category='deprecated',
             nodemodule='openalea.caribu.deprecated_nodes',
             nodeclass='S2v',
             inputs=[{'interface': None, 'name': 'CaribuScene', 'value': None,
                      'desc': ''},
                     {'interface': IInt, 'name': 'Number of layers', 'value': 5,
                      'desc': ''},
                     {'interface': IFloat, 'name': 'ZmaxLayer', 'value': 2.0,
                      'desc': ''},
                     {'interface': IBool, 'name': 'Sleep', 'value': False,
                      'desc': ''}],
             outputs=[{'interface': IStr, 'name': 'SailScene', 'desc': ''},
                      {'interface': ISequence, 'name': 'TriangleLayerLabels',
                       'desc': ''},
                      {'interface': IStr, 'name': 'Log', 'desc': ''}],
             widgetmodule=None,
             widgetclass=None,
             )
__all__.append('S2v_S2v')

Canestra = Fa(uid="5cc6256c4e8411e6bff6d4bed973e64a",
              name='Canestra',
              authors='M. Chelle,C. Fournier (wralea authors)',
              description='Nested radiosity illumination of a 3D Scene ',
              category='deprecated',
              nodemodule='openalea.caribu.deprecated_nodes',
              nodeclass='Canestra',
              inputs=[{'interface': None, 'name': 'CaribuScene',
                       'value': None, 'desc': ''},
                      {'interface': IStr, 'name': 'SailFluxes',
                       'value': None, 'desc': ''},
                      {'interface': IBool,
                       'name': 'No Multiple Scattering', 'value': True,
                       'desc': ''},
                      {'interface': IFloat, 'name': 'Sphere Diameter',
                       'value': 0.5, 'desc': ''},
                      {'interface': IBool, 'name': 'keep FF in Scene',
                       'value': False, 'desc': ''}],
              outputs=[{'interface': IStr, 'name': 'CaribuScene',
                        'desc': ''},
                       {'interface': IStr, 'name': 'Etri', 'desc': ''},
                       {'interface': IStr, 'name': 'Eabs', 'desc': ''},
                       {'interface': IStr, 'name': 'Log', 'desc': ''}],
              widgetmodule=None,
              widgetclass=None,
              )
__all__.append('Canestra')

addSoil = Fa(uid="5cc6256d4e8411e6bff6d4bed973e64a",
             name='addSoil',
             authors='C. Fournier',
             description='',
             category='deprecated',
             nodemodule='openalea.caribu.deprecated_nodes',
             nodeclass='addSoil',
             inputs=[{'interface': IData, 'name': 'CaribuScene', 'value': None,
                      'desc': ''}, {'interface': IFloat, 'name': 'altitude (z)',
                                    'value': 0.0},
                     {'interface': IBool, 'name': 'Copy Caribuscene',
                      'value': True, 'desc': ''}],
             outputs=[{'interface': IData, 'name': 'CaribuScene', 'desc': ''},
                      {'interface': IDict, 'name': 'Soil_id2Caribu_id',
                       'desc': 'mapping of soil id to internal caribu id'}],
             widgetmodule=None,
             widgetclass=None,
             )
__all__.append('addSoil')

addShapes = Fa(uid="5cc6256e4e8411e6bff6d4bed973e64a",
               name='addShapes',
               authors='C. Fournier',
               description='',
               category='deprecated',
               nodemodule='openalea.caribu.deprecated_nodes',
               nodeclass='addShapes',
               inputs=[{'interface': None, 'name': 'CaribuScene', 'value': None,
                        'desc': ''},
                       {'interface': None, 'name': 'Shapes', 'value': None,
                        'desc': ''},
                       {'interface': None, 'name': 'Tesselator', 'value': None,
                        'desc': 'PlantGL tesselator instance or None'},
                       {'interface': None, 'name': 'CanLabels', 'value': None,
                        'desc': 'Labels used by Caribu to make the association between opticals properties and shapes '},
                       {'interface': IBool, 'name': 'auto generates Can labels',
                        'value': True,
                        'desc': 'uses opt=1 and opak = 0 for all primitives, see label encode for alternatives'},
                       {'interface': IBool, 'name': 'Copy Caribuscene',
                        'value': True,
                        'desc': 'uncheck only if you know what you do (pass by reference)'}],
               outputs=[{'interface': None, 'name': 'CaribuScene', 'desc': ''},
                        {'interface': IDict, 'name': 'Shape_id2Caribu_id',
                         'desc': 'mapping of shapes id to internal caribu id'}],
               widgetmodule=None,
               widgetclass=None,
               )
__all__.append('addShapes')

output_by_id = Fa(uid="5cc6256f4e8411e6bff6d4bed973e64a",
                  name='Output by id',
                  authors='C. Fournier',
                  description='',
                  category='deprecated',
                  nodemodule='openalea.caribu.deprecated_nodes',
                  nodeclass='output_by_id',
                  inputs=[
                      {'interface': None, 'name': 'CaribuScene', 'value': None,
                       'desc': ''},
                      {'interface': None, 'name': 'Caribu Outputs',
                       'value': None, 'desc': ''},
                      {'interface': None, 'name': 'user_id -> caribu_id dict',
                       'value': None,
                       'desc': 'if given uses user id instead of caribu internal id as a key'},
                      {'interface': None,
                       'name': '(caribu or user)_id -> group_id dict',
                       'value': None,
                       'desc': 'if given uses group id instead of caribu internal id as a key'},
                      {'interface': IBool, 'name': 'aggregate', 'value': True,
                       'desc': 'should results (one per triangle) be aggregated by objects ?'}],
                  outputs=[{'interface': IDict, 'name': 'Output dict',
                            'desc': 'keys:can_id,values = variable'}],
                  widgetmodule=None,
                  widgetclass=None,
                  )
__all__.append('output_by_id')

mtg_updateMTG = Fa(uid="5cc625704e8411e6bff6d4bed973e64a",

                   name='updateMTG',
                   authors='M. Chelle,C. Fournier (wralea authors)',
                   category='deprecated',
                   nodemodule='openalea.caribu.deprecated_nodes',
                   nodeclass='updateMTG',
                   inputs=[
                       {'interface': None, 'name': 'mtg', 'value': None,
                        'desc': ''},
                       {'interface': None, 'name': 'caribu output',
                        'value': None, 'desc': ''},
                       {'interface': None, 'name': 'mtg id for triangles',
                        'value': None},
                       {'interface': IStr, 'name': 'prefix for mtg property',
                        'value': None}],
                   outputs=[
                       {'interface': None, 'name': 'updated mtg', 'desc': ''}],
                   widgetmodule=None, widgetclass=None, )
__all__.append('mtg_updateMTG')

mtg_to_canestra = Fa(uid="5cc625714e8411e6bff6d4bed973e64a",

                     name='MtgToCan',
                     authors='M. Chelle,C. Fournier (wralea authors)',
                     description='Give a canestra repr of MTG',
                     category='deprecated',
                     nodemodule='openalea.caribu.deprecated_nodes',
                     nodeclass='to_canestra',
                     inputs=[
                         {'interface': None, 'name': 'MTG', 'value': None,
                          'desc': ''},
                         {'interface': IStr, 'name': 'Property to use as OptId',
                          'value': 'optical_specie', 'desc': ''},
                         {'interface': IStr,
                          'name': 'Property to use as opacity',
                          'value': 'transparency', 'desc': ''},
                         {'interface': IStr,
                          'name': 'Property to use as Geometry',
                          'value': 'geometry', 'desc': ''},
                         {'interface': IInt, 'name': 'Deault for optid',
                          'value': 1, 'desc': ''},
                         {'interface': IInt, 'name': 'Defalt for opacity',
                          'value': 0,
                          'desc': ''},
                         {'interface': IFloat(min=0, max=16777216,
                                              step=0.000001),
                          'name': 'minimal area', 'value': 1e-05, 'desc': ''}],
                     outputs=[{'interface': None, 'name': 'mtgId',
                               'desc': 'list of mtd id associated with triangles'},
                              {'interface': None, 'name': 'CanString',
                               'desc': 'Canestra input file'}],
                     widgetmodule=None, widgetclass=None)
__all__.append('mtg_to_canestra')
