// redefinit les fonction acos et asin pour eviter les sorties de domaines
// MC99

#ifndef _Mmath
#define _Mmath
#include <cmath>
#include "ferrlog.h"

// acos
inline  double Macos(double x){
  if(fabs(x)>1.){
    if(fabs(x)-1e-5>1.){
Ferr <<"\n ** Error: Macos(x) with 1<x="  << x<<"\n" ;
      abort();
    }
    if(x>0) return 0;
    return M_PI;
  }
  return acos(x);
}//Macos
//asin 
inline  double Masin(double x){
  if(fabs(x)>1.){
    if(fabs(x)-1e-5>1.){
Ferr <<"\n ** Error: Msin(x) with 1<|x|="  << x<<"\n" ;
      abort();
    }
    if(x>0) return  M_PI/2.;
    return -M_PI/2.;
  }
  return asin(x);
}//Masin

//sqrt 
inline  double Msqrt(double x){
  if(x<0){
    if(x<-1e-5){
      Ferr <<"\n ** Error: Msqrt(x) with 0>x="  << x<<"\n" ;
      abort();
    }
    return 0.;
  }
  return sqrt(x);
}//Msqrt

#endif
