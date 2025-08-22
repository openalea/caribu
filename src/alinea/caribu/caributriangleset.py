from alinea.caribu.display import generate_scene

import numpy

class AbstractCaribuTriangleSet:
    def __init__(self):
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

    def items(self):
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
    def __init__(self, pointtuplelistdict):
        AbstractCaribuTriangleSet.__init__(self)
        self._values = pointtuplelistdict
        import itertools
        self.allpoints = list(itertools.chain(*self._values.values()))
        self.bbox = None

    def getBoundingBox(self):
        if self.bbox is None:
            x, y, z = map(numpy.array, zip(*map(lambda x: zip(*x), self.allpoints)))
            self.bbox = (x.min(), y.min(), z.min()), (x.max(), y.max(), z.max()) 
        return self.bbox     
        
    def triangle_areas(self):
        """ compute mean area of elementary triangles in the scene """

        def _surf(triangle):
            a, b, c = tuple(map(numpy.array, triangle))
            x, y, z = numpy.cross(b - a, c - a).tolist()
            return numpy.sqrt(x ** 2 + y ** 2 + z ** 2) / 2.0

        return numpy.array(list(map(_surf, self.allpoints)))

    def getZmin(self):
        return self.getBoundingBox()[0][2]

    def getZmax(self):
        return self.getBoundingBox()[1][2]

    def __getitem__(self, shapeid):
        """ Return all triangles of a shape """
        return self._values[shapeid]

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._values.keys()

    def values(self):
        return self._values.values()

    def items(self):
        return self._values.items()

    def allvalues(self, copied=False):
        from copy import copy
        if copied : 
            return copy(self.allpoints)
        else:
            return self.allpoints
    
    def allids(self):
        return self.repeat_for_triangles(self._values.keys())

    def repeat_for_triangles(self, values):
        return [v for v,nb in zip(values,[len(v) for v in self._values.values()]) for j in range(nb)]

    def getNumberOfTriangles(self, shapeid):
        return len(self._values[shapeid])

    def generate_scene(self, colorproperty):
        return generate_scene(self._values, colorproperty)

