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
from pathlib import Path


def data_path(filename):
    datadir = Path(__file__).parent / 'data'
    return datadir / filename
