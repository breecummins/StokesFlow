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

def loadPickle(basename,basedir):
    F = open(basedir+basename+'.pickle', 'r')
    mydict = cPickle.Unpickler(F).load()
    F.close()
    try:
        os.mkdir(basedir+basename)
    except:
        pass
    return mydict

