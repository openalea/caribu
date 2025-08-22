from . import Sky

class mergeNskies:
    """  Doc... """ 

    def __init__(self):
        pass


    def __call__(self, ls_sky):
        SkyRg = Sky.Sky(ls_sky[0].sec[0], ls_sky[0].sec[1])
        for i in range(1,len(ls_sky)):
            SkyRg += ls_sky[i]

        return SkyRg

