import cPickle, os

def loadPickle(basename,basedir):
    F = open(basedir+basename+'.pickle', 'r')
    mydict = cPickle.Unpickler(F).load()
    F.close()
    try:
        os.mkdir(basedir+basename)
    except:
        pass
    return mydict

