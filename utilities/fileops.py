#Created by Breschine Cummins on June 20, 2012.

# Copyright (C) 2012 Breschine Cummins
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later 
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA
#

import cPickle, os

def loadPickle(basename,basedir,newfolder='y'):
    if not basename.endswith('.pickle'):
        basename = basename+'.pickle'
    if newfolder == 'y':
        try:
            os.mkdir(os.path.join(basedir,basename.split('.pickle')[0]))
        except:
            pass
    F = open(os.path.join(basedir,basename), 'r')
    mydict = cPickle.Unpickler(F).load()
    F.close()
    return mydict

class ExtractDict():
    def __init__(self,mydict):
        self.__dict__.update(mydict)
