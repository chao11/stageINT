# check files in the directory

import os
import os.path as op
import commands

root_dir = '/hpc/crise/hao.c/data'
subjectlist = os.listdir(root_dir)

def raw_anat(path):
    n = 0
    list = os.listdir(path)
    for i in list:
        if op.splitext(i)[1]=='.gz':
            n += 1
            #print i
    return n;

"""
def raw_dwi(path):
    n = 0
    list = os.listdir(path)
    if op.isfile('%s/bvecs' % path) and op.isfile('%s/bvecs' % path):
"""

for subjectCode in subjectlist:
     path = "%s/%s/raw_anat" % (root_dir,subjectCode)
     n = raw_anat(path)

     if n!=3:
         print subjectCode+" not comleted"