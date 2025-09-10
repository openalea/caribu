""" Defines messages for deprecated nodes
"""


def GenOutput(etri, eabs):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning('This node is deprecated, use Caribu instead')


def FileCaribuScene(scene, light, pattern, opt):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning('This node is deprecated, use CaribuScene instead')


def ObjCaribuScene(scene, light, pattern, opt, waveLength):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning('This node is deprecated, use CaribuScene instead')


def getEi(cs):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is deprecated, use getIncidentEnergy instead')

def MCSail(sail_scene, sleep):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is deprecated, mcsail/sailscene wrapper code has been droped from package in july 2016')

def S2v(caribuscene, nlayers, zlayer, sleep):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is deprecated, s2v wrapper code has been droped in july 2016')

def Canestra(caribuscene, sailfluxes, direct, s_sphere, keepFF):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is deprecated, canestra wrapper code has been droped in july 2016')

def addSoil(caribuscene, zsoil, copyscene):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is deprecated, use soil options in CaribuScene instead')

def addShapes(caribuscene,shapes,tesselator, canlabels, autocan, copyscene):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is deprecated, use scene input to CaribuScene instead')

def output_by_id(caribuscene,output,mapid,groups,aggregate):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is deprecated, Caribu output are now already aggregated')

def to_canestra(g, OptId, Opak, Geometry, defopt, defopak, epsilon):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is deprecated, Caribu input accept PlantGL scene')

def updateMTG(g,canout,tri_mtgid,prefix) :
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is partly deprecated, as Caribu output are properties. '
        'For automated update of light on a mtg, see/contribute to  astk')

def PARaggregators(caribu_outdict):
    """ This is a deprecated node, not functional anymore"""
    raise DeprecationWarning(
        'This node is deprecated, as it is no more compatible '
        'with current caribu outputs. To be redeveloped based on material')

