
import os
import os.path as op
import commands

# correct the bvec:
root_dir = '/hpc/crise/hao.c/sanlm'
subjectlist = os.listdir(root_dir)
nbr= len(subjectlist)

subjectCode = "EMN02"

for subjectCode in subjectlist:
	path = "%s/%s" % (root_dir,subjectCode)

	cmdBET = 'fsl5.0-bet %s/sanlm_co* %s/T1_brain  -f 0.3 -g -0.02 -R -m ' %(path,path)
	print cmdBET
	commands.getoutput(cmdBET)
