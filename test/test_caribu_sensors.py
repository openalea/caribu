import openalea.plantgl.all as pgl

from alinea.caribu.CaribuScene import CaribuScene
from alinea.caribu.data_samples import data_path

def create_grid_sensors(dxyz, nxyz) :     
    """creates a grid of virtual sensors in Caribu and PlantGL format

    :param dxyz: length and width of a sensor and height of a layer
    :type dxyz: list of 3 float elements
    :param nxyz: number of sensors in each direction xyz
    :type nxyz: lsit of 3 int elements
    :return: virtual sensors in a Caribu triangulation format and a scene PlantGL
    :rtype: dict, pgl.Scene()
    """           
    # PlantGL scene which will stores the sensors
    s_capt = pgl.Scene()
    
    # template sensor: square upside oriented
    points = [(0, 0, 0), (0, dxyz[1], 0), (dxyz[0], dxyz[1], 0), (dxyz[0], 0, 0)]  
    normals = [(0, 0, 1) for i in range(4)]
    indices = [(0, 1, 2, 3)]
    square = pgl.QuadSet(points, indices, normals, indices)
    
    # generate the sensors in plantGL format among the grid
    ID_capt = 0
    dico_translat = {}
    for ix in range(nxyz[0]):
        for iy in range(nxyz[1]):
            for iz in range(nxyz[2]):
                # translation vector
                tx = ix * dxyz[0]
                ty = iy * dxyz[1]
                tz = iz * dxyz[2]

                # save the vector
                dico_translat[ID_capt] = [tx, ty, tz]

                # ajoute un voxel à la scene des capteurs
                sensor = pgl.Translated(geometry=square, translation=(tx, ty, tz))
                s_capt.add(pgl.Shape(geometry=sensor, id=ID_capt))
                ID_capt += 1

    
    # sensors in CARIBU scene format with triangles
    caribu_sensors = {}
    for c in s_capt:
        # Preparation
        pt_lst = c.geometry.geometry.pointList
        idx_lst = c.geometry.geometry.indexList

        # two triangles for each squarred sensor
        for i in range(0, len(idx_lst)):
            triangles = []
            
            # upper triangle
            x = [pt_lst[idx_lst[i][j]][0] + dico_translat[c.id][0] for j in range(3)]
            y = [pt_lst[idx_lst[i][j]][1] + dico_translat[c.id][1] for j in range(3)]
            z = [pt_lst[idx_lst[i][j]][2] + dico_translat[c.id][2] for j in range(3)]
            triangles.append((list(zip(x, y, z))))

            # lower triangle
            ids = [0, 2, 3]
            x = [pt_lst[idx_lst[i][j]][0] + dico_translat[c.id][0] for j in ids]
            y = [pt_lst[idx_lst[i][j]][1] + dico_translat[c.id][1] for j in ids]
            z = [pt_lst[idx_lst[i][j]][2] + dico_translat[c.id][2] for j in ids]
            triangles.append((list(zip(x, y, z))))
            
            # save the triangles
            caribu_sensors[c.id] = triangles
    
    return caribu_sensors, s_capt

if __name__ == "__main__":
    print("--- test virtual sensors")

    # inputs from example data
    debug = True
    can = data_path('filterT.can')
    sky = data_path('zenith.light')
    opts = [data_path('par.opt'), data_path('nir.opt')]

    # generation of a grid of sensors
    # size of one sensor
    dxyz = [1., 1., 1.]
    # numbers of sensors in xyz direction
    nxyz = [10, 10, 10]
    # generate the sensors
    sensors, sensors_plantgl = create_grid_sensors(dxyz, nxyz)

    # if you want to visualize the sensors
    # pgl.Viewer.display(sensors_plantgl)

    # compute radiosity
    c_scene = CaribuScene(scene=can, light=sky, opt=opts, debug = debug)
    raw, aggregated = c_scene.run(direct=True, infinite=False, sensors=sensors)


    print("--- done")
