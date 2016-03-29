
import os
import os.path as op
import commands

# correct the bvec:
root_dir = '/hpc/crise/hao.c/data'
subjectlist = os.listdir(root_dir)
nbr= len(subjectlist)

subjectCode = "EMN02"

#for subjectCode in subjectlist[30:nbr]:
path = "%s/%s/raw_dwi" % (root_dir,subjectCode)

cmdBET = 'fsl5.0-bet %s/data %s/nodif_brain  -f 0.2 -g 0 -m ' %(path,path)
print cmdBET
commands.getoutput(cmdBET)
