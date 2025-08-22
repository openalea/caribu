############ G Louarn - adaptation de spitters.c EGC grignon
from math import *

def DecliSun(DOY):
    """ Declinaison (rad) du soleil en fonction du jour de l'annee """
    alpha = 2 * pi * (DOY - 1) / 365
    return (0.006918 - 0.399912 * cos(alpha) + 0.070257 * sin(alpha))

def DayLength(latitude, decli):
    """ photoperiode (radians) en fonction de latitude (degre) et declinaison du soleil (rad) """
    lat = radians(latitude)
    d = acos(-tan(decli) * tan(lat))
    if d < 0:
        d = d + pi    
    return 2 * d

def dH(angleH):
    """ duration (hour) from  daylength angle (radians)"""
    return 24 / (2 * pi) * angleH
    
def extra(Rg, DOY, heureTU, latitude):
    """ rayonnement extraterrestre horarire """
    hrad = 2 * pi / 24 * (heureTU - 12)
    lat = radians(latitude)
    dec = DecliSun (DOY)
    costheta = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(hrad)
    Io = 1370 * (1 + 0.033 * cos(2 * pi * (DOY - 4) / 366))#eclairement (w/m2) a la limitte de l'atmosphere dans un plan perpendiculaire aux rayons du soleil, fonction du jour
    So = Io * costheta #eclairement dans un plan parallele a la surface du sol
    return So

def RdRsH(Rg, DOY, heureTU, latitude):
    """ fraction diffus/Global en fonction du rapport Global(Sgd)/Extraterrestre(Sod)- pas de temps horaire """
    hrad = 2 * pi / 24 * (heureTU - 12)
    lat = radians(latitude)
    dec = DecliSun(DOY)
    costheta = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(hrad)
    Io = 1370 * (1 + 0.033 * cos(2 * pi * (DOY - 4) / 366))#eclairement (w/m2) a la limitte de l'atmosphere dans un plan perpendiculaire aux rayons du soleil, fonction du jour
    So = Io * costheta #eclairement dans un plan parallele a la surface du sol
    RsRso = Rg / So
    R = 0.847 - 1.61 * costheta + 1.04 * costheta * costheta
    K = (1.47 - R) / 1.66
    
    if (RsRso <= 0.22) :
        return(1)
    elif (RsRso <= 0.35) :
        return(1 - 6.4 * (RsRso - 0.22)**2)
    elif (RsRso <= K) :
        return(1.47 - 1.66 * RsRso)
    else:
        return(R)



class spitters_horaire:
    """  Doc... """ 

    def __init__(self):
        pass


    def __call__(self, Tab_Rg, latitude):
        """ calcule RdRg et ajoute le resultat dans dans Tab_Rg """
        for group in range(len(Tab_Rg)):
            Tab_Rg[group].append(["RdRg"])
            for i in range(1,len(Tab_Rg[group][0])):
                Rg,DOY,heureTU = float(Tab_Rg[group][2][i]),int(Tab_Rg[group][0][i]),float(Tab_Rg[group][1][i])
                frac = self.RdRsH (Rg,DOY,heureTU,latitude)
                Tab_Rg[group][4].append(str(frac))
        
            Tab_Rg[group][3],Tab_Rg[group][4] = Tab_Rg[group][4],Tab_Rg[group][3]

        return (Tab_Rg,)
    
    def DecliSun (self,DOY):
        """ Declinaison (rad) du soleil en fonction du jour de l'annee """
        return DecliSun(DOY)    

    def DayLength (self,latitude,decli):
        """ photoperiode en fonction de latitude (degre) et declinaison du soleil (rad) """
        return DayLength(latitude,decli)
        
    def extra (self,Rg,DOY,heureTU,latitude):
        """ rayonnement extraterrestre horarire """
        return extra(Rg,DOY,heureTU,latitude)
        
    def RdRsH (self,Rg,DOY,heureTU,latitude):
        """ fraction diffus/Global en fonction du rapport Global(Sgd)/Extraterrestre(Sod)- pas de temps horaire """
        return RdRsH(Rg,DOY,heureTU,latitude)
#
# Formule pour avoir Rg horraire a partir de Rg journalier (Kaplanis, 2005.Renewable energy, 31:781:790)
#
#

def RgH (Rg,hTU,DOY,latitude) :
    """ compute hourly value of Rg at hour hTU for a given day at a given latitude
    Rg is in J.m-2.day-1
    latidude in degrees
    output is J.m-2.h-1
    """
    dec = DecliSun(DOY)
    lat = radians(latitude)
    pi = 3.14116
    a = sin(lat) * sin(dec)
    b = cos(lat) * cos(dec)
    Psi = pi * Rg / 86400 / (a * acos(-a / b) + b * sqrt(1 - (a / b)^2))
    A = -b * Psi
    B = a * Psi
    RgH = A * cos (2 * pi * hTU / 24) + B
    # Note that this formula works for h beteween hsunset eand hsunrise
    hsunrise = 12 - 12/pi * acos(-a / b)
    hsunset = 12 + 12/pi * acos (-a / b)
    return RgH
    
