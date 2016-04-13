import os
import os.path as op
import commands


root_dir = '/hpc/crise/hao.c/data'
subjectList = os.listdir(root_dir)

for subject in subjectList:
    subject_dir = op.join(root_dir,subject)
    fs_seg_dir = op.join(subject_dir,'freesurfer_seg')
   # if not op.isdir(fs_seg_dir):
    #    os.mkdir(fs_seg_dir)

    cmd_lh_seed = """fsl5.0-fslmaths %s/parc_freesurfer/parcellisation_freesurfer.nii  -uthr 11134 -thr 11134 %s/lh_STG.nii.gz
fsl5.0-fslmaths %s/parc_freesurfer/parcellisation_freesurfer.nii  -uthr 11174 -thr 11174 %s/lh_STS.nii.gz
cd %s
fsl5.0-fslmaths lh_STG.nii.gz -add lh_STS.nii.gz lh_STS+STG.nii.gz
fsl5.0-fslmaths lh_STS+STG.nii.gz -bin lh_STS+STG.nii.gz""" %(subject_dir,fs_seg_dir,subject_dir,fs_seg_dir,fs_seg_dir)
   # print cmd_lh_seed
    commands.getoutput(cmd_lh_seed)

    cmd_rh_seed = """fsl5.0-fslmaths %s/parc_freesurfer/parcellisation_freesurfer.nii  -uthr 12134 -thr 12134 %s/rh_STG.nii.gz
fsl5.0-fslmaths %s/parc_freesurfer/parcellisation_freesurfer.nii  -uthr 12174 -thr 12174 %s/rh_STS.nii.gz
cd %s
fsl5.0-fslmaths rh_STG.nii.gz -add rh_STS.nii.gz rh_STS+STG.nii.gz
fsl5.0-fslmaths rh_STS+STG.nii.gz -bin rh_STS+STG.nii.gz""" %(subject_dir,fs_seg_dir,subject_dir,fs_seg_dir,fs_seg_dir)
    #print cmd_rh_seed
    commands.getoutput(cmd_rh_seed)

    cmdtarget = """cd %s/
fsl5.0-fslmaths parc_freesurfer/parcellisation_freesurfer.nii -thr 0 -uthr 8 freesurfer_seg/sub1.nii.gz
fsl5.0-fslmaths parc_freesurfer/parcellisation_freesurfer.nii -thr 14 -uthr 16 freesurfer_seg/sub2.nii.gz
fsl5.0-fslmaths parc_freesurfer/parcellisation_freesurfer.nii -thr 24 -uthr 24 freesurfer_seg/sub3.nii.gz
fsl5.0-fslmaths parc_freesurfer/parcellisation_freesurfer.nii -thr 28 -uthr 47 freesurfer_seg/sub4.nii.gz
fsl5.0-fslmaths parc_freesurfer/parcellisation_freesurfer.nii -thr 60 -uthr 85 freesurfer_seg/sub5.nii.gz
fsl5.0-fslmaths freesurfer_seg/sub1.nii.gz -add freesurfer_seg/sub2.nii.gz -add freesurfer_seg/sub3.nii.gz -add freesurfer_seg/sub4.nii.gz -add freesurfer_seg/sub5.nii.gz -add freesurfer_seg/lh_STS.nii.gz -add freesurfer_seg/lh_STG.nii.gz -add freesurfer_seg/rh_STG.nii.gz -add freesurfer_seg/rh_STS.nii.gz freesurfer_seg/sub_target.nii.gz
fsl5.0-fslmaths parc_freesurfer/parcellisation_freesurfer.nii -sub freesurfer_seg/sub_target.nii.gz freesurfer_seg/target_mask.nii.gz""" %subject_dir
   # print cmdtarget
    commands.getoutput(cmdtarget)

    print "%s done" %subject