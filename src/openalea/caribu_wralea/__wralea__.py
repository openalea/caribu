from openalea.core import Factory as Fa
from openalea.core import (IInt, IFloat, IBool, IStr, IEnumStr, IDict,
                           ISequence, IFileStr, IFunction, IData)

__name__ = 'openalea.caribu'

__editable__ = True
__description__ = ' Caribu package '
__license__ = ''
__url__ = ''
__alias__ = ['Caribu', 'openalea.caribu.workflow']
__version__ = '0.0.4'
__authors__ = 'M. Chelle,C. Fournier, C. Pradal'
__institutes__ = 'INRA'
__icon__ = 'Caribou.png'

__all__ = []

CaribuScene = Fa(uid="96612da84e8311e6bff6d4bed973e64a",
                 name='CaribuScene',
                 authors='C. Fournier',
                 description='instantiate a CaribuScene object',
                 category='scene.light',
                 nodemodule='openalea.caribu.CaribuScene_nodes',
                 nodeclass='newCaribuScene',
                 inputs=[dict(name="in1"), dict(name="in2"), dict(name="in3"), dict(name="in4")],
                 outputs=[dict(name='obj')])

__all__.append('CaribuScene')

runCaribu = Fa(uid="96612da94e8311e6bff6d4bed973e64a",
               name='Caribu',
               authors='C. Fournier',
               description='',
               category='scene.light',
               nodemodule='openalea.caribu.CaribuScene_nodes',
               nodeclass='runCaribu',
               inputs=[dict(name="in1")],
               outputs=[{'name': 'CaribuScene'},
                        {'name': 'aggregated output'},
                        {'name': 'raw (per triangle) output'}],
               widgetmodule=None,
               widgetclass=None, )

__all__.append('runCaribu')

vcaribu = Fa(uid="96612daa4e8311e6bff6d4bed973e64a",
             name='vcaribu',
             authors='M. Chelle,C. Fournier (wralea authors)',
             description='Visualea interface to Caribu class',
             category='scene.light',
             nodemodule='openalea.caribu.caribu_shell',
             nodeclass='vcaribu',
             inputs=[
                 {'interface': IFileStr, 'name': 'canopy', 'value': None,
                  'desc': "file '.can' or file content representing the 3d scene"},
                 {'interface': IFileStr, 'name': 'lightsource', 'value': None,
                  'desc': 'file or file content describing light sources'
                          ' (direction, irradiance)'},
                 {'interface': None, 'name': 'optics', 'value': None,
                  'desc': 'list of files/files contents defining optical property'},
                 {'interface': IFileStr, 'name': 'pattern', 'value': None,
                  'desc': 'file/file content that defines a domain to till the scene'},
                 {'interface': None, 'name': 'options', 'value': None,
                  'desc': 'dictionarry allowing to set user option.'}],
             outputs=[{'interface': None, 'name': 'irradiances', 'desc': ''},
                      {'interface': None, 'name': 'Settings', 'desc': ''}],
             widgetmodule=None,
             widgetclass=None, )

__all__.append('vcaribu')

generate_scene = Fa(uid="96612dab4e8311e6bff6d4bed973e64a",
                    name='generate scene',
                    authors='C. Fournier, C Pradal',
                    description='generate a PlantGL scene form a Caribu Scene',
                    category='scene',
                    nodemodule='openalea.caribu.CaribuScene_nodes',
                    nodeclass='generate_scene_node',
                    inputs=[{'interface': None,
                             'name': 'CaribuScene'},
                            {'interface': IDict,
                             'name': 'colors',
                             'value': None}], outputs=[
        {'interface': None, 'name': 'PlantGL scene'}],
                    widgetmodule=None,
                    widgetclass=None, )

__all__.append('generate_scene')

selectOutput = Fa(uid="96612dac4e8311e6bff6d4bed973e64a",
                  name='selectOutput',
                  authors='M. Chelle,C. Fournier (wralea authors)',
                  description='Select a variable in the output dictionnary of caribu',
                  category='Unclassified',
                  nodemodule='openalea.caribu.CaribuScene_nodes',
                  nodeclass='selectOutput',
                  inputs=[
                      {'interface': None, 'name': 'output', 'value': None,
                       'desc': ''},
                      {'interface': IEnumStr(
                          enum=['area', 'Eabs', 'Ei', 'Ei_inf', 'Ei_sup']),
                       'name': 'variable', 'value': 'Ei',
                       'desc': 'see details on help tab'},
                      {'interface': IStr, 'name': 'band', 'value': None}],
                  outputs=[
                      {'interface': None, 'name': 'selected output',
                       'desc': ''},
                      {'interface': None, 'name': 'key', 'desc': ''}],
                  widgetmodule=None, widgetclass=None, )

__all__.append('selectOutput')

getIncidentEnergy = Fa(uid="96612dad4e8311e6bff6d4bed973e64a",
                       name='getIncidentEnergy',
                       authors='C. Fournier',
                       category='scene.light',
                       nodemodule='openalea.caribu.CaribuScene_nodes',
                       nodeclass='getIncidentEnergy',
                       inputs=[{'interface': None, 'name': 'CaribuScene'}],
                       outputs=[
                           {'interface': IFloat, 'name': 'Qi',
                            'desc': 'incident light flux received on an horizontal surface (m-2)'},
                           {'interface': IFloat, 'name': 'Qem',
                            'desc': 'sum of light fluxes normal to sources (m-2)'},
                           {'interface': IFloat, 'name': 'Einc',
                            'desc': 'total incident energy received on the domain '
                                    '(Einc = Qi * domain_area), or None if pattern is not set'}],
                       widgetmodule=None,
                       widgetclass=None, )

__all__.append('getIncidentEnergy')

getSoilEnergy = Fa(uid="96612dae4e8311e6bff6d4bed973e64a",
                   name='getSoilEnergy',
                   authors='C. Fournier',
                   category='scene.light',
                   nodemodule='openalea.caribu.CaribuScene_nodes',
                   nodeclass='getSoilEnergy',
                   inputs=[{'interface': None, 'name': 'CaribuScene'}],
                   outputs=[
                       {'interface': IFloat, 'name': 'Qi',
                        'desc': 'incident light flux received on soil surface (m-2)'},
                       {'interface': IFloat, 'name': 'Einc',
                        'desc': 'total energy received on soil domain '
                                '(Einc = Qi * domain_area), or None if pattern is not set'}],
                   widgetmodule=None,
                   widgetclass=None, )

__all__.append('getSoilEnergy')

WriteCan = Fa(uid="96612daf4e8311e6bff6d4bed973e64a",
              name='WriteCan',
              authors='M. Chelle,C. Fournier (wralea authors)',
              description='',
              category='io',
              nodemodule='openalea.caribu.CaribuScene_nodes',
              nodeclass='WriteCan',
              inputs=[
                  {'interface': None, 'name': 'CaribuScene', 'value': None,
                   'desc': ''},
                  {'interface': IFileStr, 'name': 'filename', 'value': None,
                   'desc': ''}],
              outputs=[
                  {'interface': IFileStr, 'name': 'filename', 'desc': ''}],
              widgetmodule=None, widgetclass=None, )

__all__.append('WriteCan')

periodise = Fa(uid="96612db04e8311e6bff6d4bed973e64a",
               name='Periodise',
               authors='M. Chelle,C. Fournier (wralea authors)',
               description='Fit the scene within its pattern',
               category='scene',
               nodemodule='openalea.caribu.CaribuScene_nodes',
               nodeclass='periodise',
               inputs=[
                   {'interface': None, 'name': 'CaribuScene'}],
               outputs=[
                   {'interface': None, 'name': 'FittedCaribuScene'}],
               widgetmodule=None,
               widgetclass=None, )

__all__.append('periodise')

encode_label = Fa(uid="96612db14e8311e6bff6d4bed973e64a",
                  name='encode label',
                  authors='C. Fournier',
                  description='',
                  category='io',
                  nodemodule='openalea.caribu.label',
                  nodeclass='encode_label',
                  inputs=[
                      {'interface': IInt, 'name': 'optical species',
                       'value': 1},
                      {'interface': IInt, 'name': 'Opacity', 'value': 1,
                       'desc': '0 means transparent(leaves), 1 means opak (stems)'},
                      {'interface': IInt, 'name': 'Plant id', 'value': 1},
                      {'interface': IInt, 'name': 'element id', 'value': 1},
                      {'interface': IInt,
                       'name': 'minimal length for the output',
                       'value': 1}],
                  outputs=[
                      {'interface': ISequence, 'name': 'canLabels',
                       'desc': ''}],
                  widgetmodule=None,
                  widgetclass=None, )

__all__.append('encode_label')

decode_label = Fa(uid="96612db24e8311e6bff6d4bed973e64a",
                  name='decode label',
                  authors='C. Fournier',
                  description='',
                  category='io',
                  nodemodule='openalea.caribu.label',
                  nodeclass='decode_label',
                  inputs=[
                      {'interface': ISequence, 'name': 'canLabels'}],
                  outputs=[
                      {'interface': ISequence, 'name': 'optical species'},
                      {'interface': ISequence, 'name': 'Opacity'},
                      {'interface': ISequence, 'name': 'Plant id'},
                      {'interface': ISequence, 'name': 'Element id'}],
                  widgetmodule=None, widgetclass=None, )

__all__.append('decode_label')

misc_reduceDict = Fa(uid="96612db34e8311e6bff6d4bed973e64a",
                     name='reduceDict',
                     authors='C. Fournier',
                     description='bind list of values of dict sharing same keys',
                     nodemodule='openalea.caribu.misc_nodes',
                     nodeclass='reduceDict',
                     inputs=[
                         {'interface': None, 'name': 'dictlist', 'value': None,
                          'desc': ''}],
                     outputs=[
                         {'interface': None, 'name': 'reduceddict',
                          'desc': ''}],
                     widgetmodule=None, widgetclass=None, )

__all__.append('misc_reduceDict')

misc_filterby = Fa(uid="96612db44e8311e6bff6d4bed973e64a",
                   name='filterby',
                   authors='C. Fournier',
                   description='Return values whose indices match condition',
                   category='Unclassified',
                   nodemodule='openalea.caribu.misc_nodes',
                   nodeclass='filterby',
                   inputs=[
                       {'interface': ISequence, 'name': 'indices',
                        'value': None, 'desc': ''},
                       {'interface': ISequence, 'name': 'values', 'value': None,
                        'desc': ''},
                       {'interface': IFunction, 'name': 'condition',
                        'value': None}],
                   outputs=[
                       {'interface': None, 'name': 'values', 'desc': ''}],
                   widgetmodule=None, widgetclass=None, )

__all__.append('misc_filterby')

misc_mydict = Fa(uid="96612db54e8311e6bff6d4bed973e64a",
                 name='mydict',
                 authors='C. Fournier',
                 description='debug dict',
                 category='Unclassified',
                 nodemodule='openalea.caribu.misc_nodes',
                 nodeclass='mydict',
                 inputs=[
                     {'interface': None, 'name': 'liste of tuple',
                      'value': None}],
                 outputs=[
                     {'interface': None, 'name': 'dict', 'desc': ''}],
                 widgetmodule=None, widgetclass=None, )

__all__.append('misc_mydict')

PARaggregators = Fa(uid="96612db64e8311e6bff6d4bed973e64a",
                    name='PARaggregators',
                    authors='M. Chelle,C. Fournier (wralea authors)',
                    description='returns a dict of aggregators (0/1) for summing Eabs at different levels',
                    category='Unclassified',
                    nodemodule='openalea.caribu.deprecated_nodes',
                    nodeclass='PARaggregators', inputs=[
        {'interface': None, 'name': 'aggregation table', 'value': None,
         'desc': ''}],
                    outputs=[
                        {'interface': None, 'name': 'aggegators', 'desc': ''}],
                    widgetmodule=None, widgetclass=None, )

__all__.append('PARaggregators')

light_source = Fa(uid="96612db74e8311e6bff6d4bed973e64a",
                  name='light source',
                  alias=['light string'],
                  authors='M. Chelle,C. Fournier (wralea authors)',
                  description=' create a monodirectional light source for caribu',
                  category='scene.light',
                  nodemodule='openalea.caribu.light',
                  nodeclass='light_source',
                  inputs=[
                      {'interface': IFloat, 'name': 'Radiance', 'value': 1,
                       'desc': ''},
                      {'interface': IFloat, 'name': 'elevation angle (deg)',
                       'value': 90,
                       'desc': ''},
                      {'interface': IFloat, 'name': 'azimuth angle (deg)',
                       'value': 0,
                       'desc': ''}],
                  outputs=[
                      {'interface': IStr, 'name': 'vector', 'desc': ''}],
                  widgetmodule=None, widgetclass=None, )
__all__.append('light_source')

# moved nodes. Re-declared here to help update by user

# ZenithPar = Fa(uid="96612db84e8311e6bff6d4bed973e64a",
#                name='CaribuZenithPar',
#                description='This node is not functional. '
#                            'Use workflow/CaribuZenithPar instead',
#                nodemodule='openalea.caribu.moved_nodes',
#                nodeclass='CaribuZenithPar',
#                category='Unclassified',
#                doc='',
#                inputs=[
#                    {'desc': '', 'interface': IData, 'name': 'CanScene',
#                     'value': None},
#                    {'desc': '', 'interface': ISequence, 'name': 'Pattern',
#                     'value': None},
#                    {'desc': '', 'interface': IEnumStr, 'name': 'scene_unit',
#                     'value': None}],
#                outputs=[
#                    {'desc': '', 'interface': None, 'name': 'CaribuScene'},
#                    {'desc': '', 'interface': None, 'name': 'CaribuOutDict'},
#                    {'desc': '', 'interface': None,
#                     'name': 'sceneid_to_caribu_id'}])

# __all__.append('ZenithPar')

# ZenithParSoil = Fa(uid="96612db94e8311e6bff6d4bed973e64a",
#                    name='CarribuZenithParSoil',
#                    description='This node is not functional.'
#                                ' Use workflow/CaribuZenithParSoil instead',
#                    nodemodule='openalea.caribu.moved_nodes',
#                    nodeclass='CaribuZenithPar',
#                    category='Unclassified',
#                    doc='',
#                    inputs=[
#                        {'desc': '', 'interface': IData, 'name': 'CanScene',
#                         'value': None},
#                        {'desc': '', 'interface': ISequence, 'name': 'Pattern',
#                         'value': None},
#                        {'desc': '', 'interface': IEnumStr, 'name': 'scene_unit',
#                         'value': None}],
#                    outputs=[
#                        {'desc': '', 'interface': None, 'name': 'CaribuScene'},
#                        {'desc': '', 'interface': None, 'name': 'CaribuOutDict'},
#                        {'desc': '', 'interface': None,
#                         'name': 'sceneid_to_caribu_id'}])

# __all__.append('ZenithParSoil')

# LIE = Fa(uid="9774ebb24e8311e6bff6d4bed973e64a",
#          name='LIE',
#          description='This node is not functional.'
#                      ' Use workflow/LIE instead',
#          nodemodule='openalea.caribu.moved_nodes',
#          nodeclass='LIE',
#          inputs=[
#              {'desc': '', 'interface': IData, 'name': 'CaribuScene',
#               'value': None},
#              {'desc': '', 'interface': None, 'name': 'EnergyDict',
#               'value': None}],
#          outputs=[
#              {'desc': '', 'interface': None, 'name': 'Efficience'},
#              {'desc': '', 'interface': None, 'name': 'Total sol'},
#              {'desc': '', 'interface': None, 'name': 'Total Incident'}])

# __all__.append('LIE')
