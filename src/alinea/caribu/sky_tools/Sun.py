#!/usr/bin/python
from math import *


class Sun:
    def __init__(self,Rsun=1.0,elev=0.,azim=0.):
        self._Rsun=Rsun
        self.elev=elev
        self.azim=azim

    def _set_Rsun(self,Rsun):
        self._Rsun=Rsun

    def _get_Rsun(self):
        return self._Rsun
    
    Rsun = property(_get_Rsun, _set_Rsun)
    
    def _set_pos_astro(self,DOY,heureTU,lat):
        dec = self.DecliSun (DOY)
        ah = self.AngleHoraire(heureTU)
        self.elev = self.SunElev(lat,dec,ah)
        self.azim = self.SunAzim(lat,dec,ah)
        
    def _get_pos_astro(self):
        return self.elev, self.azim
    
    def __str__(self): #appeler quand 'print obj'
        return 'Rsun: '+str(self.Rsun)+' elev: '+str(self.elev)+' azim: '+str(self.azim)

    def DecliSun (self,DOY):
        """ Declinaison (rad) du soleil par rapport a l'equateur en fonction du jour de l'anneee """
        alpha=2*3.14*(DOY-1)/365
        return (0.006918-0.399912*cos(alpha)+0.070257*sin(alpha))
    
    def AngleHoraire(self,heureTU):
        return 2*pi/24*(heureTU-12)
    
    def SunElev(self,lat,dec,ah):
        """ angle d'elevation en fonction de la latitude, decli et de l'angle horaire (rad); 0 = zenith; pi/2 = horizon"""
        return (3.14/2)-(asin(sin(lat)*sin(dec)+cos(lat)*cos(dec)*cos(ah))) 
       
    def SunAzim(self,lat,dec,ah):
        """ azimut du soleil en fonction de la latitude, decli et de l'angle horaire (rad) ; Nord = 0, Est = pi/2 """
        a1 = sin(lat)*cos(ah)-cos(lat)*tan(dec)
        #az = atan(sin(ah)/a1)
        #if a1<0:
        #    az=-az
        az = atan2(sin(ah),a1)
               
        # a1 = asin(sin(lat)*sin(dec)+cos(lat)*cos(dec)*cos(ah+3.14/2))
        # a2 = sin (lat)*sin(a1)-sin(dec) # agregado el 24-10 por J Prieto
        # az = acos(a2/(cos(lat)*cos(a1)))
        return -az

    def toLight(self):
        """ string representation of Sun for caribu light file """
        lightstring = ''
        if (self.elev > 0):
            dir=[0,0,0]
            dir[0] = dir[1] = sin(self.elev)
            dir[0] *= cos(self.azim)
            dir[1] *= sin(self.azim)
            dir[2] = -cos(self.elev)
            lightstring = ' '.join([str(self.Rsun)]+[str(dir[i]) for i in  range(0,3)])
        return lightstring


    def test(self):
        Su = Sun()
        print(Su)
        Su.Rsun=4.
        print(Su)
        DOY,heureTU,lat=200,11,45
        Su._set_pos_astro(DOY,heureTU,lat)
        print(Su)
        Su._get_pos_astro()
        

if __name__ == "__main__":
    import unittest
    
    class TestSimple(unittest.TestCase):
        def setUp(self):
            #import numpy as np
            self.lat=45
            self.delai=2
            self.DOY=200
            
            pass
        def tearDown(self):
            pass

        def test00_SunAzim(self):
            Su = Sun()
            Su._set_pos_astro(self.DOY,self.delai,self.lat)
            az1=Su.azim
            Su._set_pos_astro(self.DOY,self.delai+1,self.lat)
            az2=Su.azim
            print("(az1,az2) = (%5.3f,%5.3f)" %(az1,az2))
            # the earth rotates ccw, so the sun's azimut must always decrease 
            self.assertTrue(az1>az2)

    unittest.main()
