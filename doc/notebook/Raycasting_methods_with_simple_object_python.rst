
Caribu with simple python object
================================

This script demonstrate how to illuminate a scene with caribu algorithms
using only simple python objects.

I. Create scene and lights
--------------------------

.. code:: python

    # define geometry of triangles points
    triangle_1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    triangle_2  = [(0, 0, 1e-5), (1, 0, 1e-5), (0, 1, 1e-5)]
    
    # define triangles list (scene)
    triangles = [triangle_1, triangle_2]
    
    # define optical property associate to triangle
    material_triangle_1 = (0.06, 0.04) # reflectance & transmitance
    material_triangle_2 = (0.06, 0.04) # reflectance & transmitance
    
    # define materials list 
    materials = [material_triangle_1, material_triangle_2]
    
    # create a zenital light
    vertical_light = (100, # horizontal irradiance of the source
                      (0, 0, -1)) # direction vector of the source
    
    # define light sources
    lights = [vertical_light]

II. Scene illumination
----------------------

II.1 Show result
~~~~~~~~~~~~~~~~

.. code:: python

    def show_result(res):
        print "Index of the triangles : ", res['index']
        print "The internal barcode : ", res['label']
        print "The individual areas of triangles : ", res['area']
        print "Surfacic density of energy : \n"
        print "\t- absorbed by the triangles : ", res['Eabs']
        print "\t- incoming on the triangles :", res['Ei']
        print "\t- incoming on the inferior face of the triangle", res['Ei_inf']
        print "\t- incoming on the superior face of the triangle", res['Ei_sup']

II.2 Raycasting
~~~~~~~~~~~~~~~

Raycasting allows to evaluate the direct illumination (first order = (no
transmitance, no reflectance), without rediffusion) of a scene

.. code:: python

    from openalea.caribu.caribu import raycasting
    
    # call raycasting algorithm for lightening the scene
    res = raycasting(triangles, materials, lights)
    
    show_result(res)


.. parsed-literal::

    Index of the triangles :  [0.0, 1.0]
    The internal barcode :  ['100001001000', '100001001000']
    The individual areas of triangles :  [0.5, 0.5]
    Surfacic density of energy : 
    
    	- absorbed by the triangles :  [0.0, 89.887321]
    	- incoming on the triangles : [0.0, 99.87480111111111]
    	- incoming on the inferior face of the triangle [0.0, 0.0]
    	- incoming on the superior face of the triangle [0.0, 99.874802]
    

II.3 Radiosity
~~~~~~~~~~~~~~

Radiosity allows to evaluate the exact illumination (all orders, with
rediffusions) of a scene

.. code:: python

    from openalea.caribu.caribu import radiosity
    
    res = radiosity(triangles, materials, lights)
    
    show_result(res)


.. parsed-literal::

    Index of the triangles :  [0.0, 1.0]
    The internal barcode :  ['100001001000', '100001001000']
    The individual areas of triangles :  [0.5, 0.5]
    Surfacic density of energy : 
    
    	- absorbed by the triangles :  [3.615814, 90.104706]
    	- incoming on the triangles : [4.017571111111111, 100.11634]
    	- incoming on the inferior face of the triangle [0.0, 0.24154]
    	- incoming on the superior face of the triangle [4.017571, 99.874802]
    

II.3 Mixed radiosity
~~~~~~~~~~~~~~~~~~~~

Mixed radiosity allows to evaluate an optimised approximative solution
of the illumination (all orders, with rediffusions) of an infinitly
reapeated scene. The optimisation consists of using radiosity in a given
neighbourhood and a turbid medium algorithm for the rest of the
rediffusion.

.. code:: python

    from openalea.caribu.caribu import mixed_radiosity
    
    # number of layers for running the turbid medium algrothm
    layers = 2
    
    # height of the canopy
    height = 1
    
    soil_reflectance = 0.2
    
    # 2D Coordinates of the domain bounding the scene for its replication.
    # (xmin, ymin, xmax, ymax)
    domain = (0, 0, 1, 1)
    
    # diameter of the sphere defining the close neighbourhood for local radiosity.
    diameter = 0.1
    
    res = mixed_radiosity(triangles, materials, lights, domain, soil_reflectance, diameter, layers, height)
    
    show_result(res)


.. parsed-literal::

    Index of the triangles :  [0.0, 1.0]
    The internal barcode :  ['100001001000', '100001001000']
    The individual areas of triangles :  [0.5, 0.5]
    Surfacic density of energy : 
    
    	- absorbed by the triangles :  [3.620033, 90.209846]
    	- incoming on the triangles : [4.022258888888889, 100.23316222222222]
    	- incoming on the inferior face of the triangle [0.0, 0.241822]
    	- incoming on the superior face of the triangle [4.022259, 99.991341]
    

