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
""" paths to module data file
"""
try:
    from path import Path
except ImportError:
    try:
        from path import path as Path
    except ImportError:
        try:
            from openalea.core.path import path as Path
        except ImportError:
            from IPython.external.path import path as Path


def data_path(filename):
    d = Path(__file__).dirname()
    fn = 'data/' + filename
    return d / fn
