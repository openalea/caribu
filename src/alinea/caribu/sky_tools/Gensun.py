from . import Sun

class Gensun:
    """  Generate sun object from astronomical data and location latitude""" 

    def __init__(self):
        pass


    def __call__(self, Rsun,DOY,heureTU,lat):
        s=Sun.Sun()
        s.Rsun=Rsun
        s._set_pos_astro(DOY,heureTU,3.14/180*lat)

        return s
