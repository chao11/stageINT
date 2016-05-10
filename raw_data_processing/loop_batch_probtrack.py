#

import os
import os.path as op
import commands

hemi = 'rh'
altas = 'wmparc'


root_dir = '/hpc/crise/hao.c/data'
SUBJECTS_DIR ="/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db"

subjectList = os.listdir(root_dir)
for subject in subjectList[1:]:


    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
    seed_path = op.join(fs_seg_dir,'%s_STS+STG.nii.gz' %(hemi))

    xfm_path = op.join(subject_dir,'freesurfer_regist','freesurfer2fa.mat')
    bedpostx_path = op.join(subject_dir,'raw_dwi.bedpostX','merged')
    mask_path = op.join(subject_dir,'raw_dwi','nodif_brain_mask')

    #target_path = op.join(fs_seg_dir,'target_mask.nii')
    target_path = op.join(fs_seg_dir,'target_mask_%s.nii'%altas)

#    output_tracto_dir = op.join(subject_dir,'tracto','{}_STS+STG_destrieux'.format(hemi.upper()))
    output_tracto_dir = op.join(subject_dir,'tracto','%s_STS+STG_%s'%(hemi.upper(),altas))

    mat_dot = op.join(output_tracto_dir,'fdt_matrix2.dot')

    # clear wrong data
    # commands.getoutput('rm -rf %s/tracto/'%output_tracto_dir)
    if not op.isfile(mat_dot):
        commands.getoutput('rm -rf %s' %output_tracto_dir)

        cmd = "frioul_batch 'fsl5.0-probtrackx2 -x %s --onewaycondition -c 0.2 -S 2000 --steplength=0.5 -P 5000 --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --xfm=%s --forcedir --opd -s %s -m %s --dir=%s --omatrix2 --target2=%s'" %(seed_path,xfm_path,bedpostx_path,mask_path,output_tracto_dir,target_path)
        print cmd
        commands.getoutput(cmd)



