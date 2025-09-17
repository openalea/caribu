/*	S4 CALCULE LA REPARTITION DANS UNE GRILLE 3D DES SURFACES DES TRIANGLES*/
/*	CONSTITUTIFS D'UNE MAQUETTE, AINSI QUE LA DISTRIBUTION D'ORIENTATION DES*/
/* 	NORMALES AUX TRIANGLES. */
/* 	Bruno Andrieu */
/* 	INRA Bioclimatologie 78850 Thiverval-Grignon */
/* 	tel #33 1 30815527  Fax #33 1 30815527 */
/* 	Email andrieu@bcgn.inra.fr */
/*	nb1 : Dans la suite on parle indifferement d'inclinaison ou d'angle zenital, ces*/
/* 	deux termes represente la meme grandeur. */
/* 	nb2 : la maquette peut comprendre plusieurs especes vegetales. */
/* 	Dans ce cas l'analyse est faite pour chacune des especes presentes. */
/* 	La variable espece est codee via le label associ� a chaque triangle */
/* 	(espece= label/10**11) */
/*	nb3 :La structure peut etre de type periodique dans le plan horizontal.*/
/* 	(typiquement une periode = un interrang). */
/*	Une maquette periodique est constituee de la repetition d'un motif elementaire.*/
/*	Dans ce cas l'ensemble des triangles est analyse pour calculer les variations*/
/*	moyennes a l'interieur du motif. Dans le plan horizontal, le motif est divise en*/
/* 	njx * njy cellules de dimension dx et dy. */
/* 	(njx*dx et njy*dy representent la periode en x et y) */
/*            ****              ****              ****              **** */
/*           ******            ******            ******            ****** */
/*          ********          ********          ********          ******** */
/*            **                **                **                **        unite de longueur*/
/*            **                **                **                **              >-----<*/
/*             **                **                **                ** */
/*                                ! ! ! ! ! ! ! ! ! ! */
/*                              >------------------<                 ! ! dy = 0.4   et  njy = 9*/
/*                                 longueur =  njy*dy */

/*              >-----------------------------------------------------< */
/*                  longueur =  yl */


/* 	nb4 : les dimensions maximales des tableaux sont definis dans le code */
/*	par des commandes parameter. Modifier ces lignes si necessaire pour augmenter*/
/* 	le nombre de cellules ou de classes d'angles. */
/* 	Donnees en entree: */
/* 	****************** */
/* 	Fichier fort.51 contenant les triangles  au format can */
/* 	(verifier :lecture non formatee ou format libre)       */
/* 	1 enregistrement par triangle  comprenant              */
/*	-type de primitive et attributs.                       */
/* On suppose que le premier attribut/1000000 = espece         */
/* 	-les coordonnees des trois sommets du triangle (reels) */
/* 	Fichier de parametres (lecture format libre)           */
/* 	sur lequel doit etre redirige l'entree standard */
/* 		ligne 1 : nji  nja */
/* 		nji:	nombre de classes d'inclinaison */
/* 		nja :	nombre de classes d'azimuth */
/* 	ligne 2 : njz  dz(njz) */
/* 		njz :	nombre de tranches d'altitude */
/* 		dz(njz):epaisseur des tranches d'altitude, numerotees du haut */
/* 			vers la bas (classe 1= la plus haute) */
/* 	ligne 3 :  xl,njx,dx,yl,njy,dy,nje */
/* 		xl et yl: dimensions totales de la maquette en x et y */
/* 		njx njy	: nombre de divisions dans un motif en x et en y */
/* 		dx, dy	: dimensions d'une cellule selon x et y */
/* 		nje	: nombre d'especes */
/* 	ligne 4 :  tx, ty, tz */
/* 		tx,ty,tz : parametre de translation des sommets tq maq. centree */
/* 	Fichier sortie: */
/* 	*************** */
/* 	Les resultats sont edites dans un fichier fort.60 comprenant */
/* 		Dimensions de la maquette */
/* 		Nombre de fois ou le motif est repete dans la maquette. */
/* 	STATISTIQUES GLOBALES DE CHAQUE ESPECE: */
/* 	Statistiques calculees sur toute la maquette: */
/* 		Surface foliaire totale et indice foliaire */
/* 		Distribution des orientations des normales aux feuilles */
/* 		(frequence en zenith et en azimuth ) */
/* 	STATISTIQUES PAR CELLULES: */
/* 	1 ligne par cellule et par espece, comprenant: jx,jy,jz,je,u,f(ji) */
/* 		jx,jy,jz: numero de cellule en x, y, et z */
/* 		je	: numero d'espece */
/* 		u	: densite volumique de surface foliaire */
/* 		f(ji)	: distribution d'inclinaison dans la cellule */
/* 			  (c'est a dire frequence en zenith) */
/* 	Parametres */
/* 	********** */
/* 	levelmax : nombre max de niveaux de subdivision d'un triangle */
/* 	njemax */
/* 	njxmax */
/* 	njymax */
/* 	njzmax */
/* 	njimax */
/* 	njamax */
/* 	nattmax */
/* 	Compilation et Execution: */
/* 	************************* */
/* 	g++ -o s4++ s4.C -I../../bibliotek -I. -lm*/
/* 	s4 <s4.par 	(s4.par etant le fichier de parametres. */
/* 	Definition des noms de variables dans le code */
/* 	********************************************* */
/* 	nje : nombres d'especes */
/* 	nji,nja :nombre de classes de zenith et d'azimuth */
/* 	njx,njy,njz: nombre de tranches en x, y, z */
/* 	dx,dy,dz(jz): �paisseur des tranches en x, y, z */
/* 	p1, p2, p3 : sommets du triangle */

/* MC avr 97 - Lecture des donnees envoyees par caribu par le shm */
/* MC fev 98 - Creation des fichiers SPECTRAL et CROPCHAR pour un caribu + auto */
/* MC jan 00 - Creation du fichier de temperature pour SailT, modifie fev00 */
/* MC jan 00 - Calcul du profil de sinT pour mcsellers */
/* MC jan 00 - Creation d'un ficier .can trie par couche et avec un 2e id couche */
/* MC mai 2011 - passage en 1D pour regler bug espece */
/* MC avr 97 -  */

#include <iostream>	//.h>
#include <fstream>	//.h>
using namespace std ;

#include <cstdio>
#include <cstdlib>

#include <cctype>


#include <cmath>
#ifndef __GNUG__
#ifndef WIN32
#include "bool.h"
#endif  
#endif

#include <system.h>
#include <T_utilitaires.h>

#include <transf.h>

#ifdef WIN32
#include <windows.h>	// Mem partag�e via CerateFileMapping/MapViewOfFile
#endif

#include <cassert>

#define NattMax 10
#define LevelMax 6 //4:3,6
// CF 2016 : directly read nje in opt files
//#define NBOPT 10

#define max(a,b) ((a>b)?a:b)
#define min(a,b) ((a<b)?a:b)
#define S2EPSILON 1E-9

//typedef double Patch[3][3]; => defini dans transf.h

/***************      Prototypes      ********************/
double lectri(signed char&,char&,int&,long [],int&,Patch&,FILE *);
int repart(Patch ,char, int);
void calcjp(Patch, int[3][3], char&);
void classe(double, int&);
// BUG multi-specie - MCoct2010 : void affect(Patch,int*);
void affect(Patch,int*, int);
double proscal(double*, double*);
void provec(double*, double*, double*);
void norme(double*, double*);
void normal(Patch&, double&, double&);
double area(Patch &T);
// CF 2016 read number of optical species in opt file
void lect_nopt(int *nopt,char *optname);
void lect_po( Tabdyn<double,3> &Tpo,int po,char *optname);
int isid(char *name);

// Secured file close
void Sfclose(FILE ** fic, int line=161) ; // HA nov 2003

/**********************************************************/

int  ji,ja,nbz=0,nbzm=0;
int nbpatc=0;
short je;
int njx,njy,njz;
//int jp[3][3];
double dx, dy, proscal_to_surf;
double *dz = nullptr,*bz = nullptr;
int i,j,k,po,npo=0;
Tabdyn<double,4>   xladia; //je, jz, theta, phi
Tabdyn<double,4> xpo;//je,jz,po,(Rf,Tf)
Tabdyn<double,3> Tpo;

double Rs[1000], Stot;
unsigned int nbtt,nbts; 
#ifndef WIN32
extern int errno;
#endif

#ifndef WIN32
#ifdef _IPC_SGI
struct shmid_ds info;
#else
// shminfo seginf;
#endif
#else
// AFAIRE ??
#endif

// Indice du premier fichier de propri�t�s optiques
#define MIN_ARGC 6
#define MIN_OPT (MIN_ARGC-1)

bool segpar=false,genopt=false;

ferrlog Ferr((char*)"s2v.log") ;

int s2v(int argc, char **argv){
  int nja,nji,nje, nje_reel=-1;
  char ntype,optname[200];
  signed  char test;
  int natt,nsom;
  int jz,il;
  double xl,yl,dxdy,zl;
  double di,da,xymaille;
  Patch *Ts = nullptr;
  Patch T;			// def. Transf.h
  long i_att[NattMax];
  FILE *fpar=nullptr, *fmlsail= nullptr, *fsail= nullptr,
    *ftri= nullptr, *fsurf= nullptr;
  Tabdyn<double,2> xlad;// je, jz
  Tabdyn<double,3> xladi;//jz, jz, ji
  Tabdyn<double ,2> disti;
  Tabdyn<double ,2> dista;
  double *xlai = nullptr;
  double *surft = nullptr, *volume = nullptr;
  int Nt,it=0, clef;
  bool gencan=true;
  double id;

#ifndef WIN32
  int shmid ;			// Id du segment de mem. partag�e
#else
  char clef_alphanum[12] ;	// version char* de la clef numerique
  HANDLE	hSharedSeg ;	// Handle du fichier mapp�
  LPVOID	lpSharedSeg ;	// pointeur LPVOID sur seg. partag�
#endif

  for(i=0;i<argc;i++)
    Ferr <<argv[i] << " " ;
  Ferr << '\n';
  if(argc>1 &&argc< MIN_ARGC){
    Ferr<<"<!> Syntax error: "<< argv[0] ;
    Ferr<<" [shm_id  nz  Dz  file.8  file_1.opt ... file_n.opt]"<<'\n';
    Ferr << "\t MC2005"<<'\n';
    return -1;      
  }
    
  // Initialisation
  Stot=nbtt=nbts=0;
  fmlsail=fopen("leafarea","w");
  fsail=fopen("out.dang","w");
  if(argc>1){// by shared memory (called by caribu)
    genopt=true;
    if(isid(argv[1])){//by seg
      Ferr<<"Scene coming through a shm_seg"<<'\n';
      clef=atoi(argv[1]);
      DecodeClefOut(&Nt, &clef, clef);
      /* Nt=clef/100;
	 clef%=100; */
      segpar=true;
#ifndef WIN32
      shmid=shmget((key_t)clef,SEGSIZE*sizeof(Patch) ,IPC_CREAT|0666);
      Ts=static_cast<Patch *>(shmat(shmid, nullptr, 0)); // NULL
#else
      sprintf ( clef_alphanum, "%d", clef) ;
      assert ((hSharedSeg = OpenFileMapping (
					     FILE_MAP_ALL_ACCESS,
					     FALSE,
					     clef_alphanum)) != NULL ) ;

      lpSharedSeg = MapViewOfFile (
				   hSharedSeg,
				   FILE_MAP_ALL_ACCESS,
				   0,
				   0,
				   SEGSIZE*sizeof(Patch) ) ;
      assert ( lpSharedSeg != NULL ) ;
      Ts = (Patch *)lpSharedSeg ;
#endif
    }else{
      segpar=false;
      Ferr<<"Scene coming through the file "<<argv[1]<<'\n';
      ftri=fopen(argv[1],"r");
      if(ftri==nullptr){
	Ferr<<"<!> Ouverture de "<<argv[1]<<" achoppee..."<<'\n';
	return -2;
      }
    }//else by file

    // Read the parameters in argv
    njz=atoi(argv[2]);
    zl=atof(argv[3]);
    volume=new double[njz];
    dz=new double[njz];
    bz=new double[njz];
    zl/=static_cast<double>(njz);
    for(i=njz-1;i>=0;i--){
      dz[i]=zl;
      if(i==njz-1)
	bz[i]=dz[i];
      else      
	bz[i]=dz[i]+bz[i+1];
      Ferr<<":: bz("<<i<<") = "<<bz[i]<<'\n';
    }
    //Horizontal pattern
    njx = njy = 1;

    // file.8
    fpar=fopen(argv[4],"r");
    if (fpar==nullptr){
      Ferr<<"<!> Ouverture de "<<argv[4]<<" impossible..."<<'\n';
      return -1;
    }
    
    fscanf(fpar,"%lf %lf %lf %lf",&dx,&dy,&xl,&yl);
    xl-=dx; dx=xl;
    yl-=dy; dy=yl;
    //printf("xl=%g, dx=%g, yl=%g, dy=%g\n",xl,dx,yl,dy);
    Ferr<<"xl="<<xl<<", dx="<<dx<<", yl="<<yl<<", dy="<<dy<<'\n';
    Sfclose(&fpar,__LINE__);
    nji=18;
    nja=1;

    gencan=true;

    // Optical properties
    genopt=true;
    npo=argc- MIN_OPT;
	// CF 2016: get number of species from first opt file
	sprintf(optname,"%s.opt",argv[MIN_OPT]);
    lect_nopt(&nje,optname);
    // nje =  NBOPT;
    Tpo.alloue(npo,nje,3);
    Tpo.maj(0);
    for(po=0;po<npo;po++){
      sprintf(optname,"%s.opt",argv[MIN_OPT+po]);
      lect_po(Tpo,po,optname);
    }//for po
    xpo.alloue(nje,njz,npo,2); xpo.maj(0); //CF 2016: xpo a 4 dim now
   
  }
  else{// by file si argc == 1 
    fpar=stdin;
    ftri=fopen("fort.51","r");
    if(ftri==nullptr){
      Ferr<<"<!> Ouverture de fort.51 achoppee..."<<'\n';
      return -2;
    }// if ftri
    Ferr<<"Lecture du fichier parametre dans fichier :"<<'\n';
    fscanf(fpar,"%d %d %d",&nji,&nja,&njz);
    printf("nji=%d, nja=%d, njz=%d \n",nji,nja,njz);
    Ferr <<"nji="<<nji <<", nja="<<nja<< ", njz="<<njz<< '\n';
    volume=new double[njz];
    dz=new double[njz];
    bz=new double[njz];
    for(i=njz-1;i>=0;i--){
      fscanf(fpar,"%lf",&(dz[i])); //printf("dz[%d]=%lf\n",i,dz[i]);
      if(i==njz-1)
	bz[i]=dz[i];
      else      
	bz[i]=dz[i]+bz[i+1];
    }// for i
    //printf("bz[0]=%lf\n",bz[0]);
    Ferr<<"bz[0]="<<bz[0]<<'\n';
    // fscanf(fpar,"%lf %d %lf",&xl,&njx,&dx);
    fscanf(fpar,"\n%lf %d %lf  %lf %d %lf  %d",&xl,&njx,&dx,&yl,&njy,&dy,&nje);
    //printf("%lf %d %lf  %lf %d %lf %d\n",xl,njx,dx,yl,njy,dy,nje);
    Ferr<<xl<<", "<<njx<<", "<<dx<<", "<<yl<<", ";
    Ferr <<njy<<", "<<dy<<", "<<nje<<'\n';
    //printf(" DEBUG: nje par lecture .par  = %d\n", nje);
  }// else by file (argc = 1 => fort.51: old use)
  
  //alloc des tableaux de resultats
  xlai= new double[nje];
  surft=new double[nje];
  for(i=0;i<nje;i++){
    xlai[i]=0;
    surft[i]=0.;
  }
  disti.alloue(nje,nji); disti.maj(0);
  dista.alloue(nje,nja); dista.maj(0);
  xlad.alloue(nje,njz); xlad.maj(0);
  xladi.alloue(nje,njz,nji); xladi.maj(0);
  xladia.alloue(nje,njz,nji,nja); xladia.maj(0);
  //calcul des var globales
  di = 90.0/static_cast<double>(nji);
  da = 360.0/static_cast<double>(nja);
  dxdy = dx*dy;
  xymaille = (xl*yl)/(njx*njy*dx*dy);
  
  /* lecture d'un triangle, calcul de sa classe d'orientation et de la cellule de chaque sommet
     Si besoin estn subsidvison recursive du T en cas d'a cheval
  */
 
  FILE *fcan = nullptr;
  double sT;
  int nbtri=0;
  // bug mc 2011?  nje=0;

  if( gencan){
    fcan=fopen("s2v.can","w");
    fsurf=fopen("s2v.area","w");
  }
  do{
    if(segpar){
      test=1;
      natt=1;
      // copie du Ts[it]-> T
      T.t=Ts[it].t;
      for(char ii=0;ii<3;ii++)
	for(char jj=0;jj<3;jj++)
	  T.P[ii][jj]=Ts[it].P[ii][jj];

      // Bug MC09:  if(T.t<0)
      if(T.t > 0)
	i_att[0]=1;// Transparent: feuille
      else
	i_att[0]=0; //Opak: Tige
      if(T.t==0) test=-1;
      je=fabs(static_cast<double>(T.t))-1;
      if(je+1>nje_reel)nje_reel=je+1;
      //printf("it=%d/%d :: T.t=%d, je=%d\n",it,Nt,T.t,je);
      it++;
      
    }else{
      id=lectri(test,ntype,natt,i_att,nsom,T,ftri);
      //printf(">dbg apres call lectri, test = %d\n", test);
      if(test==1){
        long opak;
	nbtri++;
	//i_att[0] = label1/1000
	je=i_att[0]/100000000-1; //je : de 0 � nje (indice tableau C)
	if(je+1>nje_reel)nje_reel=je+1;
	// Bug MC09 T.t=je+1;
	T.t= - je+1; // default OPak
	if(je==-1) test=-2;//Sol
	opak= i_att[0] - i_att[0]/1000*1000;
	// dbug: printf("i_att=%ld, iatt/1e3=%ld \n",i_att[0],i_att[0]/1000);
	// MC09 i_att[0]=i_att[0]%1000;
	i_att[0]=T.t=(opak>0)?1:0;
	//if(i_att[0]>0) T.t= -T.t;
	// i_att[0] = 1 => Transparent: feuille
	// i_att[0] = 0 => Opak: Tige
	// printf("> s2v: lectri() => id:%.0f, opak=%ld,  i_att[0]=%ld,  je=%d, T.t=%d\n",id, opak,i_att[0], je, T.t);

      }
    }//else segpar

    //debug MC2011
    //  printf(">>>> Apres lcture fichier fort.51 nje = %d, nje_reel =%d\n", nje, nje_reel) ;
    if(test>-1){
      // fprintf(stderr,"label=%d, esp=%d, nbid=%d ,nbs=%d\n",i_att[0],je,natt,nsom);
      if((je<0)&&(je>=nje)&&(natt<=0)&&(natt>NattMax)&&(nsom!=3)){
	fprintf(stderr,"*** Incorect data format in fort.51 file -> break\n");
	exit(-1);
      }
      
      // feuille ou tige pour le coef de surface
      proscal_to_surf=0.5;
      if(i_att[0]==0)
	proscal_to_surf=0.25;
      // Orientation discrete du triangle
      double incl,azi;
      // for(i=0;i<3;i++)for(j=0;j<3;j++) printf("T(%d,%d) = %lf\n",i,j,T[i][j]);

      normal(T,incl,azi); 
      ji = static_cast<int>(incl / di);
      ja = static_cast<int>(azi / da);
      //ji = min(nji-1,max(0,ji));
      //ja = min(nja-1,max(0,ja));
      ji = min(nji-1, ji);
      ja = min(nja-1,ja);

      // printf("inc=%lf, azi=%lf => ji=%d, ja=%d\n",incl,azi,ji,ja);
      //calcul recursif des coord discretes du T et du cas cheval
      //printf("Repart called 1, je=%d\n", je);
      // for(int q=0;q<3;q++) printf("Patch(%d,2)=%g\n",q,T.P[q][2]);
      if(je>=0){
	il=repart(T,0, je);
        //printf(">>>> apres call repart:  je =%d \n",je);
      }
      if(gencan){
	if(segpar) {
	  id=static_cast<int>(T.t);
	}
	sT=area(T);
	fprintf(fsurf,"%.0lf\t %d\t %g\n",id,il,sT);
	fprintf(fcan,"p 2 %.0lf %d  3 ",id,il);
	for(char ii=0;ii<3;ii++)
	  for(char jj=0;jj<3;jj++)
	    fprintf(fcan,"%lf ", T.P[ii][jj]);
	fprintf(fcan,"\n");

      }
    }//if triangle et non sol

    if(test==-2 && gencan){//cas du sol

 
      //printf("==>test=%d\n",test);
      sT=area(T);
      fprintf(fsurf,"0 \t\t 999\t %g\n",sT);
      fprintf(fcan,"p 2 0 999  3 ");
      for(char ii=0;ii<3;ii++)
	for(char jj=0;jj<3;jj++)
	  fprintf(fcan,"%lf ", T.P[ii][jj]);
      fprintf(fcan,"\n");
    }
    //printf("> dbg: juste avant while(!segpar && feof(ftri): cptr = %d\n", cptr++); 

  }while((!segpar && !feof(ftri)) || (segpar && (it<Nt)) );
  //printf("> dbg: juste apres while(!segpar && feof(ftri)\n"); 

  printf("\n***  nbtri=%d, nbpatch=%d\n\n",nbtri, nbpatc);

  if(segpar){
#ifndef WIN32
    shmdt(reinterpret_cast<void *>((size_t) shmid)); //ShMDetach
#else
    UnmapViewOfFile(lpSharedSeg) ; // invalidation du ptr sur mem partagee
    // Ts = NULL ;		   // Tester avant
    CloseHandle(hSharedSeg) ;	   // Fermeture du fichier mapp�
#endif
    Ferr<< "-> Fin de lecture du segment partage: "<<it;
    Ferr<<" Triangles " << (int)nbtt<< '\n';
    Ferr<<"=> Calcul des distributions"<<'\n';
  }else{
    Sfclose(&ftri,__LINE__);
    Ferr<<__FILE__<<":"<<__LINE__<<" -> Fin de lecture de fichier\n";
    Ferr <<"=> Calcul des distributions"<<'\n';
  }
  printf(">>>> Apres lcture fichier fort.51 nje = %d, nje_reel =%d\n", nje, nje_reel);
  Ferr<<"==> nje rel = "<<nje<<'\n';
  /* Calcul des distributions spatiales de surface 
     foliaire et des frequences d'angle 
  */
  if(nbtt<1)
    Ferr<< __FILE__<<":"<<__LINE__<<"<!> Error No triangle dealt"<<'\n';
  else{
    for (jz=0; jz<njz ; jz++) {
      volume[jz] = xymaille * dxdy * dz[jz];
    }
    printf("> debut boucle nje =%d\n\n", nje);fflush(stdout); fflush(stderr); 
    double tmp;
    /* Inversion de l'ordre des boucles en mettant je en 1er pour eviter les cas
       ou une couche ne contient pas une espece et pour faire des calculs
       especes confondues - MC98 */
    int ii=0;
    // bug mai 2011
    // jx=jy=0; // jy n'avait pas l'air d'etre init et il n'y a pas de bocle sur x et y
    for (jz=0; jz<njz; jz++) {
      for (je=0; je <nje; je++) {
	for (ji=0; ji<nji; ji++) {
	  for (ja=0; ja<nja; ja++) {
	    if(jz==1){
	      // dbug:  printf(">> (jz=%d, je=%d, ji=%d) \n",jz,je,ji);
	      fflush(stdout);fflush(stderr);
	    }
	    tmp=xladia(je,jz,ji,ja);
	    xladi(je,jz,ji) += tmp;
	    
	    disti(je, ji) += tmp;
	    dista(je, ja) += tmp;
	    surft[je] += tmp;
	  }//for ja
	  xlad(je,  jz) +=xladi(je, jz, ji);
	  //   xladi(je,  jz, ji)=ii++; //dbug mc 2010
	   // dbug: printf(">> xladi(jz=%d, je=%d, ji=%d) =%.0lf\n",jz,je,ji, xladi(je,  jz, ji));fflush(stdout);fflush(stderr);
	}//for ji
	/* 	  passage aux frequences d'angles.... */
	if (xlad(je,jz) > 0) {
	  tmp=xlad(je,  jz);
	  for (ji=0; ji < nji; ji++) {
	    xladi(je, jz, ji) /= tmp;  //bug fix� - mai 2011
	  }//for ji
	}//if xlad>0 
      }//for je
    }//for jz
    // dbug:    printf("big loop :  subT=%lf, surft=%lf\n", nbts/(double)nbtt,surft[0]);fflush(stdout);


    //dbug
    for (jz=0; jz<njz && false; jz++) 
      for (je=0; je <nje; je++) 
	for (ji=0; ji<nji; ji++) {
	  printf("<<<  xladi(jz=%d, je=%d, ji=%d) =%.0lf\n",jz,je,ji, xladi(je,  jz, ji));
	}



    /* 	Calcul des distributions globales d'orientation des feuilles */
    for (je=0; je<nje; je++) {
      for (ja=0; ja<nja; ja++) {
	if(surft[je]>0)
	  dista(je, ja)  /=  surft[je];
      }
      for (ji=0; ji<nji; ji++) {
	if(surft[je]>0)
	  disti(je, ji)  /= surft[je];
      }
    }//for je
    /* 	Calcul de l'indice foliaire */
    for (je=0; je<nje; je++) {
      xlai[je]  = surft[je] / (xl * yl);
    }
    
    /* 	Editions */
    Ferr<<"=> Ecriture des resultats : (c|std)err, leafarea, out.dang"<<'\n';
    // stdout
    if(nbz>0) 
      Ferr<<"Il y a eu "<<nbz<<" depassement en z+"<<'\n';  
    if(nbzm>0) 
      Ferr<<"Il y a eu "<<nbzm<<" depassement en z-"<<'n';  
    
    //printf("xl=%lf, yl=%lf, xymaille=%lf\n\n",xl,yl,xymaille);
    Ferr<<"xl="<<xl<<", yl="<<yl<<", xymaille="<<xymaille<<'\n'<<'\n';
    Ferr<<"STATISTIQUES GLOBALES DE CHAQUE ESPECE"<<'\n';
    for (je=0; je<nje; je++) {
      //printf("esp %d : surfT=%lf - Stot=%lf, LAI=%lf\ndist d'inclinaison :",je+1, surft[je],Stot,xlai[je]);
      Ferr<<"esp "<<je+1<<" : surfT="<<surft[je]<<" - Stot="<<Stot;
      Ferr<<", LAI="<<xlai[je]<<" dist d'inclinaison :" ;  
      for (ji=0; ji<nji; ji++) {
	//printf("%lf ",disti(je,ji));
	Ferr<<disti(je,ji)<<" ";
      }
      //printf("\ndist. d'azimut : ");
      Ferr<<"\ndist. d'azimut : ";
      for (ja=0; ja<nja; ja++) {
	//printf("%lf ",dista(je,ja));
	Ferr << dista(je,ja);
      }
      Ferr << '\n';
    }
    // genere le fichier out.dang => entree de sailM pour calculer la BRDF
    Ferr << "genere le fichier out.dang => entree de sailM pour calculer la BRDF"<<'\n' ;
    //if (fsail == NULL){Ferr <<"! le FILE *fsail est NULL"<<'\n'; //HA }

    fprintf(fsail,"%lf\n",xlai[0]);
    for (ji=0; ji<nji; ji++) 
      fprintf(fsail,"%lf ",disti(0,ji));
    //genere stat par cellule et leafarea
    double t = 0.;
    printf("\n STATISTIQUES PAR CELLULE\n");
    printf("\n nje=%d,  njz=%d, nji=%d\n",nje,njz,nji);
    printf("\n# iz  iesp  LAD distrib azimuth");

    for (je=0; je <nje; je++) {  
      for (jz=0; jz<njz; jz++) {
	t += xlad(je,jz);
	if(xlad(je,jz)>0.){ // -1000 MCoct2010 debug
	  printf("\n %d  %d \t %7.6lf\t ",jz+1,je+1,(xlad(je,jz)/volume[jz]));
	  for(ji=0;ji<nji;ji++)
	    printf("%7.6lf  ",xladi(je,jz,ji));
	}
      }
    }
    
    printf("\n\n");
    
    //genere le fichier mlsail.env necessaire a  canestra 
    double Uz,x,xx;

    for (jz=0; jz<njz; jz++) {
      Uz=0;
      fprintf(fmlsail,"  %d  %d  ",0,jz+1);
      for (je=0; je <nje; je++) {
	Uz+=xlad(je,jz);// u(jz) densite vol. de LAI
      }//for je
      for(ji=0;ji<nji;ji++){
	xx=0;
	for (je=0; je <nje; je++) {
	  xx+=xladi(je,jz,ji)*xlad(je,jz);
	}//for j

	// Note: Div / 0 !
	if ((Uz<S2EPSILON) && (Uz >-S2EPSILON)) {
	  Uz = Uz<0 ? -S2EPSILON : S2EPSILON ;
	  Ferr << "\t: xx= "<< xx<<" ; Uz= "<<Uz<<'\n';
	}
	fprintf(fmlsail,"%lf ",xx/Uz);

      }//for ji

      fprintf(fmlsail,"0 0 %lf\n",Uz/volume[jz]*dz[jz]);
    }//for jz


    if(genopt){
      double Rf, Tf;
      //Genere les fichiers SPECTRAL necessaire a MCsail
      //printf(">dbg: genere fichier spectral\n");
      for(po=0;po<npo;po++){
	sprintf(optname,"%s.spec",argv[MIN_OPT+po]);
	fpar=fopen(optname,"w");
	fprintf(fpar,"%d\n%.3lf\n",njz,Rs[po]);
	for (jz=0; jz<njz; jz++) {
	  Rf=Tf=x=0;
	  for (je=0; je<nje; je++) {
	    //printf("lai(%d,%d)=%g ",je,jz,xlad(je,0,0,jz));
	    if(xlad(je,jz)>0){
	      xx=xlad(je,jz);// /(double)nje;
	      x+=xx;
	      Rf+=xpo(je,jz,po,0);
	      Tf+=xpo(je,jz,po,1);
	      //printf("=> po(je=%d/%d)=%g, Rfz=%g, Rft=%g\n",je,nje,xpo(je,0,0,jz,po,0),xpo(je,0,0,jz,po,0)*100./xx, Rf*100./x);
	      //printf("      => xx=%g,x=%g\n",xx,x);
	    }
	    //else printf(" pas de p.o. car lai nul\n");
	  }//for je
	  if(x>0){
	    Rf*=100./x;
	    Tf*=100./x;
	  }
	    
	  fprintf(fpar,"%.3lf %.3lf\n",Rf,Tf);
	}
	Sfclose(&fpar,__LINE__);
	//Genere le fichier CROPCHAR degenere necessaire a MCsail
	fpar=fopen("cropchar","w");
	fprintf(fpar,"%d\n %d %lf\n",nji,njz,dz[0]);
	Sfclose(&fpar,__LINE__);
      }//for po
    }//if genopt
  }//else pas de triangles a traiter
  // Clean up
  if(gencan){
    Sfclose(&fcan,__LINE__);
    Sfclose(&fsurf,__LINE__);
  }
  Sfclose(&fsail,__LINE__);
  Sfclose(&fmlsail,__LINE__);

  if (xlai!=nullptr)
    delete [] xlai ;
  else 
    Ferr <<__LINE__<<" : ptr Null"<<'\n';

  if ( surft != nullptr)
    delete [] surft;
  else
    Ferr <<__LINE__<<" : ptr Null"<<'\n';
      
  if (dz != nullptr)
    delete [] dz;
  else
    Ferr <<__LINE__<<" : ptr Null"<<'\n';

  if (volume != nullptr)
    delete [] volume;
  else
    Ferr <<__LINE__<<" : ptr Null"<<'\n';

  if (bz != nullptr)
    delete [] bz;
  else
    Ferr <<__LINE__<<" : ptr Null"<<'\n';

  Tpo.free();
  disti.free();
  dista.free();
  xlad.free();
  xladi.free();
  xladia.free();
  xpo.free();

  Ferr.close();
  return 0;
}//main()


/*************************
*******  lectri()  *******
**************************/
double  lectri(signed char &test,char &ntype,int &natt,long i_att[],int &nsom,Patch&T,FILE *fichier){
  char  fin;
  int ent;
  double it,p0,p1,p2,lab;
  
  //  Ferr <<"==> lectri() : DEBUT"<<'\n';
  test=fscanf(fichier, "%c",&fin);//ntype);
  ntype=fin;
  if(ntype!='p') test=-10;
 if(test==-10){ return -1;}
  fscanf(fichier, "%d", &natt);
  for (i=0; i<natt; i++) {
    fscanf(fichier, "%lf", &it);
    i_att[i]=static_cast<long>(it / 1000);
    if(i==0) {
      lab=it;
      if(lab<0){ test=-10;  return -1;}
    }
  }
  fscanf(fichier, "%d",&ent);
  nsom=ent;
  for(i=0;i<3;i++){
    fscanf(fichier, "%lf%lf%lf", &p0, &p1, &p2);
    T.P[i][0]=p0;  T.P[i][1]=p1;  T.P[i][2]=p2; 
    //printf("Can(%d,2)=%g\n",i,T.P[i][2]);
  }
  do {
    fscanf(fichier, "%c", &fin);
  } while(fin!='p' && !feof(fichier));
  if(!feof(fichier)) {
    fseek(fichier,-static_cast<long>(sizeof(char)),SEEK_CUR);
    //fputc(fin, fichier);
  } // if

/* debug 
  if(0){
    // printf("fichier=%d \n", fichier);
    printf("ntype=%c natt=%d\n", ntype, natt);
    for (i = 0; i < natt; i++)
      printf("iatt(%d) = %lf \n",i, (double)i_att[i]);
    printf("nsom=%d\n", nsom);
    printf("a=%f %f %f \n",    T.P[0][0],  T.P[0][1],  T.P[0][2]);
    printf("b=%f %f %f \n",    T.P[1][0],  T.P[1][1],  T.P[1][2]);
    printf("c=%f %f %f \n",    T.P[2][0],  T.P[2][1],  T.P[2][2]);
  }
*/
  test=1;
  //  Ferr <<"==> lectri() : FIN"<<'\n';
  return lab;
}// lectri()

/**************************************************************************/
int repart(Patch T,char level, int je){
  Patch exT,ssT;
  char acv;
  int jp[3][3];
  double G[3];
  int il=-10,iln;
  //Ferr<<" => repart called"<<'\n';
  //  Ferr<<"\t** debut repart "<<'\n';
  //printf(">>> repart(): DEBUT\n");
  level++;
  for(i=0;i<3;i++)
    for(j=0;j<3;j++){
      exT.P[i][j]=T.P[i][j];
    }
  exT.t=T.t;
  calcjp(exT,jp,acv);
  if (acv==0) {
    //  T appartient a une seule cell    
    //printf("Non ACV: T appartient a une seule cell  \n");

   G[2]=(exT.P[0][2]+exT.P[1][2]+exT.P[2][2])/3.;
   if(G[2]>=0){
     //printf(">dbg Avant call affect...\n");
     affect(exT,jp[0], je);
     il=jp[0][2];
     //printf("Non ACV: level = %d\n",(int)level);
    //if(level>1){nbtt++;nbts++;}
   }else{     
     
     nbzm++;
     il= -2;     
     //printf("else Non ACV: il = %d, nbzm=%d\n",(int)il, nbzm);

   }
  }else{
    if(level==LevelMax){
      // niveau subdiv max atteint => on stocke dans la cell de G
      //printf("ACV mais levalMax: level = %d\n",(int)level);
      for(i=0;i<3;i++)
	G[i]=(exT.P[0][i]+exT.P[1][i]+exT.P[2][i])/3.;
      if(G[2]>=0){ 
	jp[0][0]=static_cast<int>(G[0] / dx);
	jp[0][1]=static_cast<int>(G[1] / dy);
	classe(G[2], jp[0][2]);
	affect(exT,jp[0],je);
	il=jp[0][2];
	// printf("...> nbt=%d\n",nbtt);
	//nbtt++;
      }else{
	nbzm++;
	il= -1;
      }
    }
    else{
      //on subdivise le T en 4
      for (j=0; j<3; j++) 
	for (i=0; i<3; i++)
	  ssT.P[j][i]=(exT.P[j][i]+exT.P[(j+1)%3][i])/2.;
      for (i=0; i<3; i++){
	T.P[0][i]=exT.P[0][i];
	T.P[1][i]=ssT.P[0][i];
	T.P[2][i]=ssT.P[2][i];
      }
      //for(int q=0;q<3;q++) printf("Sub1(%d,2)=%g\n",q,T.P[q][2]);
      iln=repart(T,level,je);
      il=max(il,iln);
      for (i=0; i<3; i++){
	T.P[0][i]=ssT.P[0][i];
	T.P[1][i]=exT.P[1][i];
	T.P[2][i]=ssT.P[1][i];
      }
      // for(int q=0;q<3;q++) printf("Sub2(%d,2)=%g\n",q,T.P[q][2]);
      iln=repart(T,level,je);
      il=max(il,iln);
      for (i=0; i<3; i++){
	T.P[0][i]=ssT.P[1][i];
	T.P[1][i]=ssT.P[2][i];
	T.P[2][i]=exT.P[2][i];
      }
      //for(int q=0;q<3;q++) printf("Sub3(%d,2)=%g\n",q,T.P[q][2]);
      iln=repart(T,level,je);
      il=max(il,iln);
      for (i=0; i<3; i++){
	T.P[0][i]=ssT.P[0][i];
	T.P[1][i]=ssT.P[1][i];
	T.P[2][i]=ssT.P[2][i];
      }
      // for(int q=0;q<3;q++) printf("Sub4(%d,2)=%g\n",q,T.P[q][2]);
      iln=repart(T,level,je);
      il=max(il,iln);
    }
  }
  //Ferr<<"\t** sortie repart "<<'\n';  
  // printf(">>> repart(): sortie il=%d\n", il);
  return il;
}//repart()


/**************************************************************************/
void calcjp(Patch T, int jp[3][3], char &acv){
  /* 	Calcul des positions en unites dx dy dz */
  /*	nb: le mode d'affectation est tel que la position (0,0,1) est centree sur (0, 0, 0.5*bz(njz))*/
  int itest12=0,itest13=0;

  nbtt++;
  //  Ferr<<"** calcjp() : Debut"<<'\n';
  //Ferr << "dx= "<<dx<<" dy= "<<dy<<'\n';
  for(i=0;i<3;i++){
    jp[i][0]=static_cast<int>(T.P[i][0] / dx);
    jp[i][1]=static_cast<int>(T.P[i][1] / dy);
    classe(T.P[i][2], jp[i][2]);
  }
  for(i=0;i<3;i++){
    itest12 += fabs(static_cast<double>(jp[0][i] - jp[1][i]));
    itest13 += fabs(static_cast<double>(jp[0][i] - jp[2][i]));
    //  printf("calcjp:  i=%d, t12=%d, t13=%d\n",i,abs(jp[0][i]-jp[1][i]),fabs(jp[0][i]-jp[2][i]));
  }
  acv = itest12 + itest13;
  //printf("** calcjp() : Fin\n");
}// calcjp()

/***********************************************/
void  classe(double z, int &jz){
  /* 	Classement de la valeur z. Les classes sont definies par leur borne */
  /* 	sup�rieures, le numero de classe est dans le sens des z decroissants. 
   */
  /* 	nb.: les z negatifs sont classes dans la classe la plus basse... */
  /* 	if (z.ge.bz(1)) print *,"subroutine class : depassement en z",z 
	retourne le no de couche
*/
  int arret=0;

  if (z >= bz[0]){ 
    nbz++;
    // printf("=> Depasst en z+: %.3lf >= %.3lf\n",z,bz[0]);
  }
 if (z <0){ 
   jz=-10;
 }else{
   /* 	if (z.lt.0.0) print *,"subroutine class : z negatif :",z */
   for (j=1; j<njz; j++) {
     if (z > bz[j]) {
       arret=1;
       break;
     }//if
   }//for
   if(arret==1)
     jz = j-1;
   else
     jz = njz-1;
 }
 // return jz;
}// classe()

/*******************************************************************/
  // BUG multi-specie - MC oct 2010
  //void affect(Patch T,int *jp){ 
void affect(Patch T,int *jp, int je){ 
  
  double a[3], b[3], c[3];
  int  jz,po;
  double  surftri;

  /*  mise a jour du tableau xladia avec un triangle dont les 3 sommets appar tiennent a une meme cellule
   */
  jz = jp[2];
  /* decalage de 1 et prise en compte de possibles coordonnees negatives */

  for (i = 0; i < 3; i++) {
    a[i] = T.P[1][i] - T.P[0][i];
    b[i] = T.P[2][i] - T.P[0][i];
  }
  provec(a, b, c);
  surftri = proscal_to_surf * sqrt(proscal(c, c));
  Stot+=surftri;
  nbpatc++;
  //printf("surftri=%lf\n",surftri);
  //printf("affect() je=%d, jx=%d, jy=%d, jz=%d, ji=%d et ja=%d\n",je,jx,jy,jz,ji,ja);
  // printf("affect() je=%d, T.t=%d, [%.2lf,%.2lf,%.2lf]\n",je,T.t,Tpo(1,fabs(T.t),0),Tpo(1,fabs(T.t),1),Tpo(1,fabs(T.t),2));
   xladia(je, jz, ji, ja) +=  surftri;
   //printf(">>> affect(): avnt call norme()\n"); 
   //CF 2016: do not compute norme if triangle area is zero
   if (surftri > 0)
	norme(c,c);
   //printf(">>> affect(): je = %d\n",je);
   if(genopt){
    for(po=0;po<npo;po++){
      //Bug MC feb 2006
      // pour caribu T.t>0 = transparent, T.t<=0 = opak !!
      //BUG  if(T.t<0){//Transparent
	 if(T.t>0){//Transparent
	xpo(je,  jz,po,0)+=surftri*Tpo(po,je,1); //Rf
	xpo(je,  jz,po,1)+=surftri*Tpo(po,je,2); //Tf
	//printf("> affect(): cas transperentt: je=%d, jz=%d :: surftri=%.3g, Tpo(%d,%d,0)=%g\n",je,jz,surftri,po,T.t,Tpo(po,je,0));
      }else{//Opak
	printf("> affect() :Cas opak: Tpo(%d,%d,0)=%g\n",po,T.t,Tpo(po,je,0));
	xpo(je,  jz,po,0)+=surftri*Tpo(po,je,0); //Rt
      }
    }//for po
  }//if segpar
}// affect()


/***********************************/
double proscal(double *a, double *b){
    double ret_val = 0;
    for (i = 0; i < 3; i++) {
      ret_val += a[i]*b[i];
    }
    return ret_val;
}// proscal()

/********************************************/
void provec(double *a, double *b, double *c){
    c[0] =  a[1] * b[2] - a[2] * b[1];
    c[1] = -a[0] * b[2] + a[2] * b[0];
    c[2] =  a[0] * b[1] - a[1] * b[0];    
}// provec()

/*****************************************************************/
void norme(double *x, double *xn){
  double xnor;
  
  xnor = sqrt(proscal(x, x));
  if (xnor < 1e-20){ 
    fprintf(stderr, "<!> s2v.cpp: norme() erreur: nombre=%g < 1e-20!! => exit(-1)\n", xnor); 
    exit(-1);
  }
  for(i=0; i<3; i++) 
    xn[i] = x[i] / xnor;
}// norme()


/***********************************************************************/
void normal(Patch &T, double &zen, double &az){
  double u, v, w, r2, r1, p3p2[3];

  //printf("normal() : Debut\n");
  for (i = 0; i < 3; i++) {
    //printf("(%d) :%lf - %lf\n",i,T.P[1][i], T.P[2][i]);
    p3p2[i] = T.P[1][i] - T.P[2][i];
  }
  u=  T.P[0][1]*p3p2[2] - T.P[0][2]*p3p2[1] + T.P[1][1]*T.P[2][2] - T.P[2][1]*T.P[1][2];
  v= -T.P[0][0]*p3p2[2] + T.P[0][2]*p3p2[0] - T.P[1][0]*T.P[2][2] + T.P[2][0]*T.P[1][2];
  w=  T.P[0][0]*p3p2[1] - T.P[0][1]*p3p2[0] + T.P[1][0]*T.P[2][1] - T.P[2][0]*T.P[1][1];
  r2 = u*u + v*v;
  if (r2 == 0) {
      zen = 0;
      az = 0;
     }
    else{
      r1 = sqrt(r2);
      zen = acos(fabs(w/sqrt(r2+w*w))) * 57.29577951;
      az = atan2(u/r1, v/r1) * 57.29577951;
      if (az < 0) {
	az += 360;
      }
    }
  // printf("normal() : Fin\n");  
} // normal()
/***********************************************************************/
double area(Patch &T){
  double a[3], b[3], c[3];
  int i;
  for (i = 0; i < 3; i++) {
    a[i] = T.P[1][i] - T.P[0][i];
    b[i] = T.P[2][i] - T.P[0][i];
  }
  provec(a, b, c);
  return  proscal_to_surf * sqrt(proscal(c, c));
} // area()


/***********************************************************************/
void synterr(char * fname,int l){
  Ferr<< "<!> Syntax error in the file "<<fname;
  Ferr<<" describing the optical properties at the line "<<l<<"\n\tMC98"<<'\n';
}

// CF 2016 : add nbopt reader
void lect_nopt(int *nopt,char *optname) {
  ifstream fopti(optname,ios::in);
  char c, line[256];
  int l=0;
  if (!fopti){
    Ferr << "<!> Error(lect_po)  unable to open "<<optname<<"\n";
    cerr.flush();
    exit (-3);
  }
  // scan fichier '.opt'
  do{
    fopti>>c; l++; 
    if(!fopti) break;
    switch(c) {
    case '#':
    case 's':
	case 'e':
      fopti.getline(line,256);		        
      break;
	case 'n':
      fopti >> *nopt;
      fopti.getline(line,256);	        
      break;
	default  :
      synterr(optname,l);
	      }//switch c
	if (c == 'n')
		break;
  } while(fopti);
  fopti.close();
}//lect_nopt()



void lect_po( Tabdyn<double,3> &Tpo,int po,char *optname){
  ifstream fopti(optname,ios::in);
  char c, line[256];
  double Rt,Rf,Tf;
  int l=0,esp=0;
  if (!fopti){
    Ferr << "<!> Error(lect_po)  unable to open "<<optname<<"\n";
    cerr.flush();
    exit (-3);
  }
  // lecture des proprietes optiques (fichier '.opt')
  do{
    fopti>>c; l++; 
    if(!fopti) break;
    switch(c) {
    case '#':
    case 'n':
      fopti.getline(line,256);		        
      break; 
    case 's':
      fopti>>c; if(c!='d') synterr(optname,l);
      fopti>>Rs[po];
      fopti.getline(line,256);
      break;
    case 'e':
      //BRDF type : d
      fopti>>c; if(c!='d') synterr(optname,l);
      Rf=Tf=0; 
      // stem reflectance
      fopti>>Rt; 
      Tpo(po,esp,0)=Rt;
      //BRDF type : d
      fopti>>c; if(c!='d') synterr(optname,l);
      // upper leaf Rf and Tf
      fopti>>Rf>>Tf;
      //BRDF type : d
      fopti>>c;  if(c!='d') synterr(optname,l);
      //lower leaf Rf => equivalent Rf
      fopti>>Rt;  Rf=(Rf+Rt)/2.;
      //lower leaf Tf => equivalent Tf
      fopti>>Rt;  Tf=(Tf+Rt)/2.;
      //printf("Rf(%d)=%.4g, Tf=%.4g\n",po,Rf,Tf);
      Ferr << "Rf("<<po<<")="<<Rf<<", Tf="<<Tf<<'\n';
      Tpo(po,esp,1)=Rf;
      Tpo(po,esp,2)=Tf;	  
      esp++;
      fopti.getline(line,256);
      break; 
    default  :
      synterr(optname,l);  
    }//switch c
  } while(fopti);
  fopti.close();
}//lect_po()
/***********************************************************************/

int isid(char *name){
  int i,l,num=1;
  
  l=strlen(name);
  Ferr<<name<<" contient "<< l <<" chars"<<'\n';
  for(i=0;i<l && num; i++){
    num=isdigit(name[i]);
  }
  return num;
}//isid()

void Sfclose(FILE ** fic, int line) {
  if (*fic == nullptr){
    Ferr << "Sfclose appele ligne "<<line<<" : fic == NULL"<<'\n';
  } else {
    //Ferr << "Sfclose appele ligne "<<line<<" : Ok "<<'\n';
    fflush (*fic);
    fclose (*fic);
  }
}
// Sfclose()
