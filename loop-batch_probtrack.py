import os
import os.path as op
import commands


root_dir = '/hpc/crise/hao.c/data'

SUBJECTS_DIR ="/hpc/banco/voiceloc_full_database/fs_5.3_sanlm_db"

subjectList = os.listdir(root_dir)
for subject in subjectList:

    print subject
    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
    seed_path = op.join(fs_seg_dir,'lh_STS+STG.nii.gz')
    target_path = op.join(fs_seg_dir,'target_mask.nii')
    xfm_path = op.join(subject_dir,'freesurfer_regist','freesurfer2fa.mat')
    bedpostx_path = op.join(subject_dir,'raw_dwi.bedpostX','merged')
    mask_path = op.join(subject_dir,'raw_dwi','nodif_brain_mask')
    output_tracto_dir = op.join(subject_dir,'tracto','LH_STS+STG_destrieux')

    cmd = "frioul_batch 'fsl5.0-probtrackx2 -x %s --onewaycondition -c 0.2 -S 2000 --steplength=0.5 -P 5000 --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --xfm=%s --forcedir --opd -s %s -m %s --dir=%s --omatrix2 --target2=%s'" %(seed_path,xfm_path,bedpostx_path,mask_path,output_tracto_dir,target_path)
    #print cmd
    commands.getoutput(cmd)
