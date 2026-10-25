import Sky

def add_sun(sky, sun):
    '''    add sun as a supplementary light source in a sky object
    '''
    sky.set_Rsun2(sun)
    return sky
