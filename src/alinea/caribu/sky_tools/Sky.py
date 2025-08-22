from math import *
from numpy import array

class Sky:
    def __init__(self,Nbp,Nbt):
        """ initialise sky avec I=0; Nbp=nb secteurs d'azimut, Nbt nombre secteurs zenithaux """
        self.sec = [Nbp,Nbt]
        self.sky = []

        self.dp = 2 * pi / Nbp
        self.dt = pi / 2 / Nbt

        da = self.dp
        dz = self.dt
        
        I=0
        for j in range(Nbp):
            for k in range (Nbt):
              azim,elv = j * da + da / 2, k * dz + dz / 2
              dir = self.Vdir(elv,azim)        
              self.sky.append([I] + dir + [j,k])

    def set_Rd(self,Rd,Tsky):
        """  I=Rd distribue selon type de ciel =soc/uoc """
        Nbp,Nbt=self.sec[0],self.sec[1]
        da,dz = self.dp,self.dt
    
        count=0
        for j in range(Nbp):
            for k in range (Nbt):
              azim,elv = j * da + da / 2, k * dz + dz / 2
              if(Tsky=='soc'):
                  I = self.soc (elv, dz, azim, da) * Rd
              else:
                  I = self.uoc (elv, dz, azim, da) * Rd

              self.sky[count][0] = I 
              count+=1

    def set_Rsun(self,sun):
        """ I=sun.Rsun;  ajoute sun dans secteur selon elv et azim """
        Nbp,Nbt=self.sec[0],self.sec[1]

        sect_sun = self.Which_SkySec(Nbp,Nbt,sun._get_pos_astro())
        for i in range(len(self.sky)):
            if self.sky[i][4]==sect_sun[0] and self.sky[i][5]==sect_sun[1]:
                self.sky[i][0]+=sun.Rsun #ajoute le soleil

    def set_Rsun2(self,sun):
        """ ajoute sun position exacte en rajoutant une ligne au ciel discretise """
        elv, azim = sun._get_pos_astro()
        dir = self.Vdir(elv,azim)
        I = sun.Rsun
        self.sky.append([I]+dir+[-1,-1])

    def __add__(self,sky2):
        """ ajoute 2 ciels ; surcharge +"""
        if self.sec!=sky2.sec:
            print("sky sectors do not match!")
        else:
            res=Sky(self.sec[0],self.sec[1])
            for i in range(len(self.sky)):
                res.sky[i][0]=self.sky[i][0]+sky2.sky[i][0]

            return res

    def get_Rs(self):
        """ calcule du rayonnement total  """
        res=0
        for i in range(len(self.sky)):
            res=res+self.sky[i][0]

        return res

    def uoc (self, teta, dt, phi, dp):
        """ teta: angle zenithal; phi: angle azimutal du soleil """
        dt /= 2.
        x = cos(teta-dt)
        y = cos(teta+dt)
        E = (x*x-y*y)*dp/2./pi
        return E
    
    def soc (self, teta, dt, phi, dp):
        """ teta: angle zenithal; phi: angle azimutal du soleil """
        dt /= 2.
        x = cos(teta-dt)
        y = cos(teta+dt)
        E = (3/14.*(x*x-y*y) + 6/21.*(x*x*x-y*y*y))*dp/pi
        return E

    def Vdir(self,elv,azim):
        """ genere vecteur direction """
        dir=[0,0,0]
        dir[0] = dir[1] = sin(elv)
        dir[0] *= cos(azim)
        dir[1] *= sin(azim)
        dir[2] = -cos(elv)
        return dir

    def Which_SkySec(self,nbp,nbt,SunPos):
        """ retourne les indices (zenith,azim) du secteur de ciel dans lequel setrouve le soleil; nbt et nbp sont les nb de secteurs zenitaux et azimutaux """
        """ retourne -1 la nuit """
        elv,azim=SunPos[0],SunPos[1]
        if elv<0:
            return [-1,-1]
        else:
            Belv=array(list(range(nbt)))*(pi/(2*nbt)) #vecteur bornes des secteurs elv
            Bazim=array(list(range(nbp)))*(2*pi/(nbp)) - pi 
            Velv=abs(Belv-elv).tolist() # liste des distances des bornes a azim
            Vazim=abs(Bazim-azim).tolist()
            Ielv=Velv.index(min(Velv)) # recupere index de la distance mini
            Iazim=Vazim.index(min(Vazim))
            return [Ielv,Iazim]

    def Noralised_Sky(self):
        """ si RS >0, copie vraie du ciel avec somme des intensites = 1 """
        Rs = self.get_Rs()
        res = Sky(self.sec[0], self.sec[1])
        

        if Rs>0.:
            for i in range(len(res.sky)):
              res.sky[i][0] = self.sky[i][0]/Rs
              
        return res


    def test(self):
        s=Sky(4,3)
        s.sky
        s.get_Rs()
        s.set_Rd(1.,'soc')
        s2=Sky(4,3)
        s2.set_Rd(3.,'soc')
        s5 = s2.Noralised_Sky()
        s5.get_Rs(), s2.get_Rs()
        s3=s+s2
        s3.get_Rs()
        s4=Sky(4,3)
        sun1=Sun()
        s4.set_Rsun(sun1)


# a finir
# quid de la boucle?
# traitements en amont + spitters
