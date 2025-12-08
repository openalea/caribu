from openalea.caribu.display import generate_scene
from openalea.libcaribu.io import read_can, decode_labels

import numpy
from itertools import chain

class AbstractCaribuTriangleSet:
    def __init__(self):
        self.bbox = None
        pass

    def getBoundingBox(self):
        pass

    def triangle_areas(self):
        pass

    def getZmin(self):
        pass

    def __getitem__(self, shapeid):
        """ Return all triangles of a shape """
        pass

    def keys(self):
        pass

    def values(self):
        pass

    def shapes(self):
        pass

    def allvalues(self, copied=False):
        pass

    def allids(self):
        pass

    def getNumberOfTriangles(self, shapeid):
        pass

    def generate_scene(self, colorproperty):
        pass

    def __len__(self):
        raise NotImplemented()

class CaribuTriangleSet(AbstractCaribuTriangleSet):
    def __init__(self, tri_dict, scales=None):
        AbstractCaribuTriangleSet.__init__(self)
        self.triangle_set = tri_dict
        all_tris = chain.from_iterable(tri_dict.values())
        self.all_triangles = numpy.fromiter(
            (coord for tri in all_tris for pt in tri for coord in pt),
            dtype=float
        ).reshape(-1, 3, 3)
        self.counts = numpy.fromiter((len(v) for v in tri_dict.values()), dtype=int)
        self.scales = {'shape': list(tri_dict.keys())}
        self.by_scale = {'shape': numpy.repeat(numpy.arange(len(self.counts)), self.counts)}
        if scales is not None:
            orgs, by_org = self.scales['shape'], self.by_scale['shape']
            for scale, mapping in scales.items():
                groups = [mapping[o] for o in orgs]
                group_names, id_to_groupid = numpy.unique(groups, return_inverse=True)
                self.scales[scale] = group_names
                self.by_scale[scale]  = id_to_groupid[by_org]

    @classmethod
    def fromfile(cls, source):
        triangles, labels = read_can(source)
        tri_dict = {}
        unique_ids, inv = numpy.unique(labels, return_inverse=True)
        specie, plant, leaf, element = decode_labels(unique_ids)
        for i, uid in enumerate(unique_ids):
            tri_dict[uid] = triangles[inv == i]
        plant_scale = dict(zip(unique_ids, plant))
        elt_scale = dict(zip(unique_ids, element))
        sp_scale = dict(zip(unique_ids, specie))
        leaf_scale = dict(zip(unique_ids, leaf))
        return cls(tri_dict, scales={'plant': plant_scale, 'specie': sp_scale, 'leaf': leaf_scale, 'elt': elt_scale})


    def getBoundingBox(self):
        if self.bbox is None:
            x, y, z = self.all_triangles.reshape(-1, 3).T
            self.bbox = (x.min(), y.min(), z.min()), (x.max(), y.max(), z.max()) 
        return self.bbox     
        
    def triangle_areas(self):
        """ compute area of elementary triangles in the scene """

        A = self.all_triangles[:, 0, :]
        B = self.all_triangles[:, 1, :]
        C = self.all_triangles[:, 2, :]

        return 0.5 * numpy.linalg.norm(numpy.cross(B - A, C - A), axis=1)


    def sum_at_scale(self, arr, scale='shape'):
            if isinstance(scale, (list, tuple)):
                invs = [self.by_scale[s] for s in scale]
                sizes = [len(self.scales[s]) for s in scale]

                mult = numpy.cumprod([1] + sizes[:-1])
                by = sum(i * m for i, m in zip(invs, mult))

                # build combined keys
                keys = [
                    tuple(self.scales[s][(i // m) % size]
                          for s, m, size in zip(scale, mult, sizes))
                    for i in range(mult[-1] * sizes[-1])
                ]

            else:
                keys = self.scales[scale]
                by = self.by_scale[scale]

            _sum = numpy.bincount(by, weights=arr, minlength=len(keys))

            return dict(zip(keys, _sum))

    def mean_at_scale(self, arr, scale=None):
        return self.sum_at_scale(arr / self.counts, scale=scale)



    def getZmin(self):
        return self.getBoundingBox()[0][2]

    def getZmax(self):
        return self.getBoundingBox()[1][2]

    def __getitem__(self, shapeid):
        """ Return all triangles of a shape """
        return self.triangle_set[shapeid]

    def __len__(self):
        return len(self.triangle_set)

    def keys(self):
        return self.triangle_set.keys()

    def values(self):
        return self.triangle_set.values()

    def shapes(self):
        return self.triangle_set.shapes()

    def allvalues(self, copied=False):
        from copy import copy
        if copied : 
            return copy(self.allpoints)
        else:
            return self.allpoints
    
    def allids(self):
        return self.repeat_for_triangles(list(self.triangle_set.keys()))

    def repeat_for_triangles(self, values):
        return numpy.repeat(values, self.counts)


    def getNumberOfTriangles(self, shapeid):
        return len(self.triangle_set[shapeid])

    def generate_scene(self, colorproperty):
        return generate_scene(self.triangle_set, colorproperty)

