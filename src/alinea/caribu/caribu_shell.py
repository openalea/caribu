# -*- python -*-
#
#       Copyright 2015 INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       WebSite : https://github.com/openalea-incubator/caribu
#
# ==============================================================================
""" Chaine les appel de s2v, mcsail et canestra, comme le fait cpfg via caribu
    Syntaxe: caribu.csh Ds file.light file.can nz h file.8 file1.opt ... fileN.opt

  Exemple /home/tld/chelle/QHS/Calgary/dev/Canestra/MC-SAIL/Test
  caribu.csh 0.2 sky2.light Tout.can 3 18 Tout.8 test nir par

  MC98
  C. Pradal - 2009
  MC09
  Contact: chelle@grinon.inra.fr
  INRA - INRIA - CIRAD
"""
import os
from subprocess import Popen, STDOUT, PIPE
import tempfile
import platform
try:
    from path import Path
except ImportError:
    try:
        from path import path as Path
    except ImportError:
        try:
            from openalea.core.path import path as Path
        except ImportError:
            from IPython.external.path import path as Path


def _process(cmd, directory, out):
    """
    Run a process in a shell.
    Return the outputs in a file or string.
    """
    # print ">> caribu.py: process(%s) called..."%(cmd)

    f = open(out, 'w')
    if platform.system() == 'Darwin':
        p = Popen(cmd, shell=True, cwd=directory,
                  stdin=PIPE, stdout=f, stderr=PIPE)
        status = p.communicate()
    else:
        p = Popen(cmd, shell=True, cwd=directory,
                  stdin=None, stdout=f, stderr=STDOUT)
        status = p.wait()

    f.close()

    # print "<<< caribu.py: process finished!"
    return status


def _safe_iter(obj, atomic_types=(str, int, float, complex)):
    """Equivalent to iter when obj is iterable and not defined as atomic.
    If obj is defined atomic or found to be not iterable, returns iter((obj,)).
    safe_iter(None) returns an empty iterator"""
    if not isinstance(obj, atomic_types):
        try:
            return iter(obj)
        except TypeError:
            pass
    return iter((obj,) * (obj is not None))


def _abrev(fnc, maxlg=1):
    """
    abbreviate a text string containing a path or a file content to the first maxlg lines,
    addind '...' when the number of libnes is greater than maxlg
    """
    if fnc is None or os.path.exists(fnc):
        return str(fnc)
    lines = fnc.splitlines()
    if maxlg <= 1:
        return lines[0] + ' ... '
    elif len(lines) <= maxlg:
        return str(fnc)
    else:
        return '\n' + '\n'.join(lines[0:maxlg]) + "\n..."


class CaribuError(Exception):
    pass


class CaribuOptionError(CaribuError):
    pass


class CaribuIOError(CaribuError):
    pass


class CaribuRunError(CaribuError):
    pass


class Caribu:
    def __init__(self,
                 canfile=None,
                 skyfile=None,
                 optfiles=None,
                 patternfile=None,
                 sensorfile=None,
                 optnames=None,
                 direct=True,
                 infinitise=True,
                 nb_layers=None,
                 can_height=None,
                 sphere_diameter=-1,
                 debug=False,
                 resdir="./Run",
                 resfile=None,
                 projection_image_size=1536
                 ):
        """
        Class fo Nested radiosity illumination on a 3D scene.

        canfile: file '.can' (or file content) representing 3d scene
        skyfile: file/file content containing all the light description
        optfiles: list of files/files contents defining optical property
        sensorfile: file or file content with virtual sensor positions
        optnames: list of name to be used as keys for output dict (if None use the name of the opt files or
        the generic names band0,band1 if optfiles are given as content)
        patternfile: file/file content that defines a domain to till the scene.
        direct: consider only direct projection
        infinitise: Consider a toric canopy (infinite). Needs a pattern to take effect
        nb_layers: number of layers to be considered for the scene
        can_height: height of the can scene
        sphere_diameter: used for the radiosity
        debug : print messages and prevent removal of tempdir
        resdir : store caribu results as files in resdir if resdir is not None, store nothing otherwise
        resfile : store caribu output dictionary in file resfile (with pickle) if resfile is not None,
        store nothing otherwise
        projection_image_size : the size (pixel) of the projection image used to compute the first order lighting
        of the scene
        """
        if debug:
            print("\n >>>> Caribu.__init__ starts...\n")
        # debug mode
        self.my_dbg = debug
        # print "my_dbg = ",   self.my_dbg
        # tempdir (initialised to allow testing of  existence in del)
        self.tempdir = Path('')

        # Input files
        self.scene = canfile
        self.sky = skyfile
        self.opticals = optfiles
        self.pattern = patternfile
        self.sensor = sensorfile

        # User options
        self.optnames = optnames
        self.resdir = resdir
        self.resfile = resfile

        self.infinity = infinitise  # consider toric canopy
        self.form_factor = None
        self.direct = direct  # direct light only
        self.nb_layers = nb_layers  # grid turbid medium
        self.can_height = can_height  # height of the canopy
        self.sphere_diameter = sphere_diameter  # parameters for nested radiosity

        self.canestra_name = "canestrad"
        self.sail_name = "mcsail"
        self.periodise_name = "periodise"
        self.s2v_name = "s2v"
        self.ready = True
        self.img_size = projection_image_size
        if debug:
            print("\n <<<< Caribu.__init__ ends...\n")

    def __del__(self):
        if self.my_dbg and self.tempdir.exists():
            print("Caribu.__del__ called, tmp dir kept: %s" % self.tempdir)
        else:
            if self.tempdir.exists():
                # print 'Remove tempfile %s'%self.tempdir
                self.tempdir.rmtree()

    def __str__(self):
        s = """
            scene %s
            sky %s
            optnames %s
            opticals %s
            pattern %s
            sensor %s
            infinity %s
            direct %s
            nb_layers %s
            can_height %s
            sphere_diameter %s
            form_factor %s
            ------------
            canestrad: %s
            mcsail: %s
            periodise: %s
            s2v: %s
        """ % (_abrev(self.scene), _abrev(self.sky), ' '.join(map(str, _safe_iter(self.optnames))),
               ''.join(map(_abrev, _safe_iter(self.opticals))), self.pattern,_abrev(self.sensor),  self.infinity, self.direct,
               self.nb_layers, self.can_height, self.sphere_diameter, self.form_factor, self.canestra_name,
               self.sail_name, self.periodise_name, self.s2v_name)
        if self.my_dbg:
            sopt = """
            -----------
            debug on
            tempdir %s
            resdir %s
            resfile %s
            """ % (self.tempdir, self.resdir, self.resfile)
            s += sopt
        return (s)

    def show(self, titre="############"):
        print("\n>>> Caribu state in ", titre)
        print(self)
        print("<<<<\n\n")

    def init(self):
        if self.scene is None or self.sky is None or self.opticals is None or self.opticals == []:
            raise CaribuOptionError(
                "Caribu has not been fully initialized: scene, sky, and optical have to be defined\n     =>  Caribu can not be run... - MC09")

        # print "infty, pattern", self.infinity, self.pattern

        if not self.direct:
            if self.sphere_diameter < 0:
                # Compute classic radioity without toric scene
                if self.infinity:
                    raise CaribuOptionError("incompatible options for radiosity : sphere_diameter < 0 && infinity")
            else:  ## diameter >=0
                # consider a toric canopy
                if not self.infinity:
                    raise CaribuOptionError(
                        "incompatible options for nested radiosity: no infinity &&  sphere_diameter >= 0 ")

        if self.pattern is None and self.infinity:
            raise CaribuOptionError('pattern not specified => Caribu cannot infinitise the scene')

        self.form_factor = True
        # self.canestra_1st = True # Boolean that indicates the first or not times, canestra is called thus form factors computed...

        # nrj is a dictionary of dictionary, each containing one simulation outputs. There will be as many dictionaries as optical files given as input
        self.nrj = {}
        # sensor measurements
        self.measures = {}

        if self.my_dbg:
            self.show("Caribu::init()")

        # name of band to process (if not given)
        if self.optnames is None:
            # try to derive from filename or use generic name
            optn = []
            for i, opt in enumerate(_safe_iter(self.opticals)):
                if os.path.exists(opt):
                    name = str(Path(Path(opt).basename()).stripext())
                    optn.append(name)
                else:
                    optn.append('band%d' % i)
            self.optnames = optn

        # Working directory
        self.setup_working_dir()

        # Copy the files (or file content) in the tempdir
        self.copyfiles()

    def init_periodise(self):
        """ init caribuscene for a periodise-only run. """
        if self.scene is None or self.pattern is None:
            raise CaribuOptionError("Periodise has not been fully initialized: scene and pattern have to be defined")
        self.infinity = True
        self.setup_working_dir()
        self.copyfiles(skip_opt=True, skip_sky=True)

    def setup_working_dir(self):
        """ Create working directories for caribu."""
        try:
            if self.my_dbg:
                self.tempdir = Path("./Run-tmp")
                if not self.tempdir.exists():
                    self.tempdir.mkdir()
            else:
                # build a temporary directory
                self.tempdir = Path(tempfile.mkdtemp())

            # Result directory (if specified)
            if self.resdir is not None:
                self.resdir = Path(self.resdir)
                if not self.resdir.exists():
                    self.resdir.mkdir()
        except:
            raise CaribuIOError(
                ">>> Caribu can't create appropriate directory on your disk : check for read/write permission")

    def copyfiles(self, skip_sky=False, skip_pattern=False, skip_opt=False):
        d = self.tempdir

        if str(self.scene).endswith('.can'):
            fn = Path(self.scene)
            fn.copy(d / fn.basename())
        else:
            fn = d / 'cscene.can'
            fn.write_text(self.scene)
        self.scene = Path(fn.basename())

        if not skip_sky:
            if os.path.exists(self.sky):
                fn = Path(self.sky)
                fn.copy(d / fn.basename())
            else:
                fn = d / 'sky.light'
                fn.write_text(self.sky)
            self.sky = Path(fn.basename())

        if not skip_pattern:
            if self.infinity:
                if os.path.exists(self.pattern):
                    fn = Path(self.pattern)
                    fn.copy(d / fn.basename())
                else:
                    fn = d / 'pattern.8'
                    fn.write_text(self.pattern)
                self.pattern = Path(fn.basename())

        if self.sensor is not None:
            if os.path.exists(self.sensor):
                fn = Path(self.sensor)
                fn.copy(d / fn.basename())
            else:
                fn = d / 'sensor.can'
                fn.write_text(self.sensor)
            self.sensor = Path(fn.basename())

        if not skip_opt:
            optn = [x + '.opt' for x in _safe_iter(self.optnames)]
            try:
                for i, opt in enumerate(_safe_iter(self.opticals)):
                    # safe_iter allows not to iterate along character composing the optfile name when only one optfile is given
                    if os.path.exists(opt):
                        # print opt
                        fn = Path(opt)
                        fn.copy(d / optn[i])
                    else:
                        fn = d / optn[i]
                        fn.write_text(opt)
                self.opticals = list(map(Path, _safe_iter(optn)))
            except IndexError:
                raise CaribuOptionError("Optnames list must be None or as long as optfiles list")

    def store_result(self, filename, band_name):
        """
        Add a new entry to the nrj dictionary, using band_name as key and a dictionary build from filename as value.
        The dictionary build from filename is organised as follows:
            - doc : the first line of filename, that contains information on the simulation
            - data : a dictionary of vectors, each containing a column of filename
            Columns are:
                - index (float): the polygon index
                - label (str): its can label
                - area (float): its area
                - Eabs,Ei_sup and Ei_inf (float): surfacic density (energy/s/m2) of, respectively, absorbed energy, irradiance on the adaxial side and irradiance on the abaxial side of polygons
        """

        f = open(filename)
        doc = f.readline()  # elimine la ligne de commentaire
        f.readline()  # elimine la ligne de commentaire
        idx = []
        label = []
        area = []
        Eabs = []
        Ei_sup = []
        Ei_inf = []
        for line in f:
            elements = line.split()  # tu split ta chaine en fonction d'une string de separation: par defaut c'est ' ', '\t', mais tu peux faire split(','), ...
            floats = [float(el) for el in
                      elements]  # tu convertis toutes les strings du tableau elements en float. floats est un tableau
            idx.append(floats[0])
            lab = elements[1]
            if len(lab) < 11:
                lab = (12 - len(lab)) * '0' + lab
            label.append(lab)
            area.append(floats[2])
            Eabs.append(floats[3])
            Ei_sup.append(floats[4])
            Ei_inf.append(floats[5])

        f.close()
        data = {'index': idx, 'label': label, 'area': area, 'Eabs': Eabs, 'Ei_sup': Ei_sup, 'Ei_inf': Ei_inf}
        self.nrj[band_name] = {'doc': doc, 'data': data}

    def store_sensor(self, filename, band_name):
        id, eio, ei, area = [], [], [], []
        with open(filename, 'r') as handle:
            for line in handle:
                elements = line.split()
                floats = [float(el) for el in elements]
                id.append(floats[0])
                eio.append(floats[1])
                ei.append(floats[2])
                area.append(floats[3])
        self.measures[band_name] = {'sensor_id': id, 'Ei0': eio, 'Ei': ei, 'area': area}

    def run(self):
        """
        The main Caribu program.
        1. Periodise: to convert the scene into an infinite one.
        2. s2v: Surface to volume based on the scene, the height of the canopy and optical prop.
        3. mcsail: mean fluxes in the canopy
        4. canestra: compute radiosity
        5. save output on disk if resfile specified
        """
        if self.my_dbg:
            print("\n >>>> Caribu.run() starts...\n")
        self.init()
        if self.infinity:
            self.periodise()
        if self.infinity and not self.direct:
            self.s2v()
            for opt in self.opticals:
                self.mcsail(opt)
        for opt in self.opticals:
            self.canestra(opt)
        if self.resfile is not None:
            import pickle
            file = open(self.resfile, 'w')
            pickle.dump(self.nrj, file)
            # To restore the value of the object to memory, load the object from the file.
            # Assuming that pickle has not yet been imported for use, start by importing it:

            # import pickle
            # file = open('caribu_run.obj', 'r')
            # caribu_run = pickle.load(file)
            # x=caribu_run
            # print x['par']['data']['Eabs'][0]
        if self.my_dbg:
            print("\n <<<< Caribu.run() ends...\n")

    def run_periodise(self):
        """ Run Periodise as a standalone program
        """
        self.init_periodise()
        self.periodise()
        d = self.tempdir
        fin = open(d / self.scene)
        canstring = fin.read()
        fin.close()
        return canstring

    def periodise(self):
        d = self.tempdir
        name, ext = self.scene.splitext()
        outscene = name + '_8' + ext
        cmd = '%s -m %s -8 %s -o %s ' % (self.periodise_name, self.scene, self.pattern, outscene)
        if self.my_dbg:
            print(">>> periodise() : ", cmd)
        status = _process(cmd, d, d / "periodise.log")
        if (d / outscene).exists():
            self.scene = outscene
        else:
            f = open(d / "periodise.log")
            msg = f.readlines()
            f.close()
            print(">>>  periodise has not finished properly => STOP")
            raise CaribuRunError(''.join(msg))

    def s2v(self):
        d = self.tempdir
        wavelength = ' '.join([fn.stripext() for fn in self.opticals])
        cmd = "%s %s %d %f %s " % (
            self.s2v_name, self.scene, self.nb_layers, self.can_height, self.pattern) + wavelength
        if self.my_dbg:
            print(">>> s2v() : ", cmd)
        status = _process(cmd, d, d / "s2v.log")
        # Raise an exception if s2v crashed...
        leafarea = d / 'leafarea'
        if not leafarea.exists():
            f = open(d / "s2v.log")
            msg = f.readlines()
            f.close()
            print(">>>  s2v has not finished properly => STOP")
            raise CaribuRunError(''.join(msg))

    def mcsail(self, opt):
        d = self.tempdir
        optname, ext = Path(opt.basename()).splitext()
        (d / optname + '.spec').copy(d / 'spectral')

        cmd = "%s %s " % (self.sail_name, self.sky)

        if self.my_dbg:
            print(">>> mcsail(): ", cmd)
        logfile = "sail-%s.log" % optname
        logfile = d / logfile
        status = _process(cmd, d, logfile)

        mcsailenv = d / 'mlsail.env'
        if mcsailenv.exists():
            mcsailenv.move(d / optname + '.env')
        else:
            f = open(logfile)
            msg = f.readlines()
            f.close()
            print(">>>  mcsail has not finished properly => STOP")
            raise CaribuRunError(''.join(msg))

    def canestra(self, opt):
        """Fonction d'appel de l'executable canestrad, code C++ compilee de la radiosite mixte  - MC09"""
        # canestrad -M $Sc -8 $argv[6] -l $argv[2] -p $po.opt -e $po.env -s -r  $argv[1] -1
        d = self.tempdir
        optname, ext = Path(opt.basename()).splitext()
        if self.my_dbg:
            print(optname)
        str_pattern = str_direct = str_FF = str_diam = str_env = str_sensor = ""

        if self.infinity:
            str_pattern = " -8 %s " % self.pattern

        if self.direct:
            str_direct = " -1 "
        else:
            str_diam = " -d %s " % self.sphere_diameter

            if self.form_factor:
                # compute formfactor
                self.form_factor = False
                self.FF_name = tempfile.mktemp(prefix="", suffix="", dir="")
                str_FF = " -f %s " % self.FF_name
            else:
                str_FF = " -w " + self.FF_name
            if self.sphere_diameter >= 0:
                str_env = " -e %s.env " % optname

        if self.sensor is not None:
            str_sensor = " -C %s " % self.sensor

        str_img = "-L %d" % self.img_size

        cmd = "%s -M %s -l %s -p %s -A %s %s %s %s %s %s %s " % (
            self.canestra_name, self.scene, self.sky, opt, str_pattern, str_direct, str_diam, str_FF, str_env, str_img, str_sensor)
        if self.my_dbg:
            print((">>> Canestrad(): %s" % cmd))
        status = _process(cmd, self.tempdir, d / "nr.log")

        ficres = d / 'Etri.vec0'
        ficsens = d / 'solem.dat'
        if ficres.exists():
            self.store_result(ficres, str(optname))

            if self.sensor is not None:
                if ficsens.exists():
                    self.store_sensor(ficsens, str(optname))

            if self.resdir is not None:
                # copy result files
                fdest = Path(optname + ".vec")
                if self.my_dbg:
                    print(fdest)
                ficres.move(self.resdir / fdest)

                if self.sensor is not None:
                    fdest = Path(optname + ".sens")
                    if self.my_dbg:
                        print(fdest)
                    ficsens.move(self.resdir / fdest)
        else:
            f = open(d / "nr.log")
            msg = f.readlines()
            f.close()
            print(">>>  canestra has not finished properly => STOP")
            raise CaribuRunError(''.join(msg))

        if (d / Path("nr.log")).exists():
            # copy log files
            fic = Path("nr-" + optname + ".log")
            (d / "nr.log").move(d / fic)

        if self.my_dbg:
            print(">>> caribu.py: Caribu::canestra (%s) finished !" % optname)


def vcaribu(canopy, lightsource, optics, pattern, options):
    """
    low level interface to Caribu class call
    Caribu allows nested radiosity illumination on a 3D scene.

    Available options are:
         1st: consider only direct projection if True.
         Nz: number of layers to be considered for the scene
         Hc: height of the can scene
         Ds: diameter of the sphere for nested radiosity
         debug : print messages and prevent removal of tempdir
         resdir : store caribu results as files in resdir
                 if resdir is not None, store nothing otherwise
         wavelength: list of name to be used as keys
                 for output dict (if None use the name of the opt files
                 or the generic names band0,band1 if optfiles are given
                 as content)
    """

    sim = Caribu(resdir=None, resfile=None)  # no output on disk
    # --canfile
    sim.scene = canopy
    # --optics
    sim.opticals = optics
    # --skyfile
    sim.sky = lightsource
    # --pattern
    sim.pattern = pattern
    # --options (if different from caribu defaults)
    if options is not None:
        # --scatter
        if '1st' in list(options.keys()):
            sim.direct = options['1st']
        # --infinity
        if 'infinity' in list(options.keys()):
            sim.infinity = options['infinity']
        # --nb_layers
        if 'Nz' in list(options.keys()):
            sim.nb_layers = options['Nz']
            # --can_height
        if 'Hc' in list(options.keys()):
            sim.can_height = options['Hc']
            # --sphere_diameter
        if 'Ds' in list(options.keys()):
            sim.sphere_diameter = options['Ds']
        # --debug mode (if True, prevent removal of tempdir)
        if 'debug' in list(options.keys()):
            sim.my_dbg = options['debug']
        # --names of optical properties (useful if opticals are given as strings
        if 'wavelength' in list(options.keys()):
            sim.optnames = options['wavelength']
        # size of the projection image for first order
        if 'projection_image_size' in list(options.keys()):
            sim.img_size = options['projection_image_size']
    status = str(sim)
    sim.run()
    irradiances = sim.nrj

    # return outputs
    return irradiances, status


def vperiodise(canopy, pattern):
    """ low level interface to periodise. return modified canopy in can format """
    sim = Caribu(resdir=None, resfile=None)  # no output on disk
    # --canfile
    sim.scene = canopy
    # --pattern
    sim.pattern = pattern
    periodic_scene = sim.run_periodise()

    return periodic_scene



def main(my_arg):
    """

    MC09
    """
    print(">>> caribu.py :main(%s) starts..." % my_arg)
    sim = Caribu()

    print(">>> caribu.py :main(): options analysis")
    # parse command line options
    import getopt
    try:
        opts, args = getopt.getopt(my_arg, "hc:s:o:p:XN:Z:D:",
                                   ["help", "canfile=", "skyfile=", "optfiles=", "pattern=", "scatter", "nb_layers=",
                                    "can_height=", "sphere_diameter="])
    except getopt.GetoptError as msg:
        print(msg)
        print("for help use --help")
        sys.exit(2)
    # process options and  arguments
    print("opts=%s (len=%s)" % (opts, len(opts)))
    print("args=%s (len=%s)" % (args, len(args)))
    sim.opticals = args
    for opt, arg in opts:
        print("opt=%s, arg=%s" % (opt, arg))
        if opt in ("-h", "--help"):
            print("caribu.py use...\n 'hc:s:p:1N:Z:D:', ['help','canfile=','skyfile=','pattern=','direct','nb_layers=','can_height=','sphere_diameter=' + optfiles (last args)")
            sys.exit(0)
        elif opt in ("-c", "--canfile"):
            sim.scene = arg
        elif opt in ("-s", "--skyfile"):
            sim.sky = arg
        elif opt in ("-p", "--pattern"):
            sim.pattern = arg
            sim.infinity = True
        elif opt in ("-X", "--scatter"):
            sim.direct = False
        elif opt in ("-N", "--nb_layers"):
            sim.nb_layers = int(arg)
        elif opt in ("-Z", "--can_height"):
            sim.can_height = float(arg)
        elif opt in ("-D", "--sphere_diameter"):
            sim.sphere_diameter = arg
        else:
            print("<!> Erreur Option non trouvee: opt=%s, arg=%s" % (opt, arg))

    print(">>> caribu.py :main():  run caribu...")
    sim.show()
    sim.run()


if __name__ == "__main__":
    import sys

    print("caribu.py called ...")
    main(sys.argv[1:])
    # USE
    # %run caribu.py  -h
    # %run caribu.py  --help
    # %run caribu.py  -8 # Error !!
    # %run caribu.py  --canfile='data/filterT.can' --skyfile='data/zenith.light' 'data/par.opt'
    # %run caribu.py  --canfile='data/filterT.can' --skyfile='data/zenith.light' 'data/par.opt' 'data/nir.opt'
    # %run caribu.py  --canfile='data/filterT.can' --skyfile='data/zenith.light'  --pattern='data/filter.8'  'data/par.opt' 'data/nir.opt'
    # %run caribu.py  --canfile='data/filterT.can' --skyfile='data/zenith.light'  --pattern='data/filter.8' -X -Z 21 -N 6 -D 10 'data/par.opt' 'data/nir.opt'

'''
# Original caribu.csh in C-shell....
# MC00

mv *.spec temp/
mv spectral temp/
mv cropchar temp/
mv leafarea temp/
mv *.env temp/
mv *.log temp/

#envt
setenv Sc /tmp/virtualis.can

#
if ( $#argv < 7 ) then
 echo "Syntax error: caribu.csh Ds file.light file.can nz h file.8 file1.opt [... fileN.opt]"
else

# Periodise (caribu le fait il ?)
#
  echo "periodise"  -m $argv[3] -8 $argv[6] -o $Sc
 periodise -m $argv[3] -8 $argv[6] -o $Sc
# S2V
 echo "s2v" $argv[3-]
 #s2v $argv[3-] >& s2v.log
 s2v $Sc $argv[4-] >& s2v.log
 if( $status != 0) then
   echo "S2V a plante!"
   exit(-1)
 endif # Loop sur le p.o.
 set i=1
    foreach po ($argv[7-])
   echo "==> "$po, $i
   ## MCSAIL
   cp $po".spec" spectral
   echo "  mcsail" $argv[2]
   mcsail $argv[2]>& sail-$po.log
   if( $status != 0) then
    echo "Sail a plante!"
    exit(-1)
   endif
   mv mlsail.env $po.env
   ## Canestra
   date
   if ( $i == 1 ) then
      ## 1ere fois: calcul des FF et Bfar
      echo "Appel no. "$i ": canestrad" -M $Sc -8 $argv[6] -l $argv[2] -p $po.opt -e $po.env -s -r  $argv[1] -1
       canestrad -M $Sc -8 $argv[6] -l $argv[2] -p $po.opt -e $po.env -s -r $argv[1]  -t /tmp/ -f caribu -v 2 >& nr-$po.log
  else
      ## ie fois: pas de calcul des FF et Bfar
      echo "Appel no. "$i ": canestrad" -M $Sc -8 $argv[6] -l $argv[2] -p $po.opt -e $po.env -s -r  $argv[1] -1
       canestrad -M $Sc -8 $argv[6] -l $argv[2] -p $po.opt -e $po.env -s -r $argv[1]    -t /tmp/ -w caribu  -v 2>& nr-$po.log
  # nr-$po'_'$i.log
  endif
  date
  cp E0.dat E_$po.dat
  cp B.dat B_$po.dat
  echo " "
 @ i = $i  + 1
 end
 # \rm leafarea spectral cropchar
endif
## FIN


### Trucs & Astuces
#
# $status = variable de retour
#
# Exemple de calcul en variable Cshell
#setenv c `cat $1|wc -w`
#@ c = $c  / $r

'''
