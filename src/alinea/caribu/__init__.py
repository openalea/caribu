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

# {# pkglts, base
from importlib.metadata import version

try:
    __version__ = version("alinea.caribu")
except Exception:
    # package is not yet installed
    pass
# #}
