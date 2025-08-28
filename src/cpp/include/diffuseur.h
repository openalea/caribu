#ifndef R_DIFFUSEUR
#define R_DIFFUSEUR

#include "primitive.h"
#include "verbose.h"
#include "actop.h"
//#include "outils.h"
// DIFFUSEUR

class Diffuseur {
protected:
  Primitive *prim;
  Actop *opti;
public:
  virtual ~Diffuseur() = default;

  static unsigned int idx;
  signed char acv;// A Cheval
  Diffuseur(){ //cout<<"Diffuseur [constructeur] Warning : no parameters!\n";
  }
  Diffuseur (Primitive *pprim,Actop *actop) : prim(pprim),opti(actop) {}
  Primitive& primi() { return *prim; }
  virtual  Vecteur normal() {return prim->normal();}
  Point centre() const {return prim->centre();}
  double surface() const {return prim->surface();}
  double name() const {return prim->name();}
  Vecteur azi0() const {return prim->azi();}//azimuth zero
  virtual bool isopaque()=0;
  virtual bool isreal()=0;

  virtual double intersect(Param_Inter parag,Point *I)
  {return prim->intersect(parag,I);}
  void  show(const char *texte=(char *)"",ostream& out=cout) const
  // montre!
  { prim->show(texte,out);  }
  virtual  unsigned int num()=0;
  virtual  void togle_face()=0;
  virtual void active(Vecteur&)=0;
  virtual void active(unsigned char)=0;
  virtual void activ_num(unsigned int)=0;
  virtual unsigned char face()=0;
  virtual double rho()=0;
  virtual double tau()=0;
  // amie
  friend int maxE(Diffuseur*,Diffuseur*); // utilise par QuickSort (TabDyn, ListeD)  
  // renvoie -1 si E1>E2, 0 si E1=E2, 1 si E1<E2 (Ei delta energie du difuseur i
  friend ostream& operator << (ostream& out,Diffuseur & diff);
  // cas du patch voir si on fait une hierachie double (class Patch)
  static int nb_patch() {return 1;}
};//Diffuseur


class DiffO :public Diffuseur{
protected:
  unsigned int no{};
public:
  DiffO() = default;
  DiffO(Primitive *p,Actop *actop): Diffuseur(p,actop) { no=idx++; }
  bool isopaque() override {return true;}
  bool isreal() override {return true;}
  double rho() override {return  opti->rho();}
  double tau() override {return opti->tau();}
  unsigned int num() override {return no;}
  void togle_face() override {}
  void active(Vecteur &dir) override {}
  void active(unsigned char cefa) override {}
  void activ_num(unsigned int nummer) override {}
  unsigned char face() override { return 0;}
};//DiffO

class DiffP :public DiffO{ //Capteur virtuel

public:
  DiffP(Primitive *p,Actop *actop): DiffO(p,actop){ }
  explicit DiffP(Primitive *p) {Actop *actop; actop=new Lambert(); prim=p; opti=actop;no=idx++;}
  bool isreal() override{return false;}
};//DiffP


class DiffT :public Diffuseur{
private:
  Face actif; // 0 : face sup - 1 : face inf
  unsigned int no[2]{};
  Actop * optback;
  Actop *popt(Face side) const {return side==sup? opti:optback;}
public:
  DiffT(Primitive *p,Actop *actop, Actop * backopt): Diffuseur(p,actop){
    optback=backopt;
    no[0]=idx++;
    no[1]=idx++;
    actif=sup;//default comme ca!
  }
  bool isopaque() override {return false;}
  bool isreal() override {return true;}
  Vecteur normal() override {
    if(actif==sup) return prim->normal();
    else           return prim->normal()*-1.0;
  }
 
  double rho() override {return  popt(actif)->rho();}
  double tau() override {return popt(actif)->tau();}
 
  unsigned int num() override {return no[actif];}
  void togle_face() override {actif =1-actif;}
  void active(Vecteur &dir) override {
    if(  dir.prod_scalaire(prim->normal())<0)
      actif=sup;
    else actif = inf;
  }
  void active(unsigned char cefa) override {
    actif=cefa==1?inf : sup;
  }
  void activ_num(unsigned int nummer) override {
    if(nummer!=num()){
      togle_face();
      if(nummer!=num()) {
	Ferr<<" Stronzo faux nummer dans DiffT::activ_num"<<'\n';
	exit(30);}
    }
  }//activ_num()
  unsigned char face() override {return actif;}
};//DiffT
#endif
