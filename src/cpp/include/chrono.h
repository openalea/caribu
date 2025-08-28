/*----------------------------------------------------------------------------
    Chrono.H	simple Chrono class (derived from Rudi's Chrono class)

    by Andreas Hohmann, ZIB, Hohmann@sc.ZIB-Berlin.DE
    
    last change:	Sep 10, 1994
    Compiler: 		CC/g++
-----------------------------------------------------------------------------*/

#ifndef _Chrono_h
#define _Chrono_h
#include <iostream>
using namespace std;

#include "ferrlog.h"

class ChronoData;

class Chrono  {
public:
  Chrono();
  ~Chrono();
  
  void Start();
  void Stop() const;
  double Seconds() const;
    
  ostream& PrintOn(ostream &s) const;
  ferrlog& PrintOn(ferrlog &s) const;
  static char* Name() { return (char*)"Chrono"; }
private:
  ChronoData *data;
  friend ostream & operator <<(ostream&,Chrono&); 
// HA 01 2001
  friend ferrlog & operator <<(ferrlog&, Chrono& ) ;
};
ostream & operator <<(ostream& out ,Chrono& uhr);
// HA 01 2001
ferrlog & operator <<(ferrlog&, Chrono& ) ;
#endif
