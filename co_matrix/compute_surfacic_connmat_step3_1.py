# hao.c
# 10/06/2016
"""
compute the surfacic connectivity matrix.
create surfacic seed mask use the connectivity matrix and compute the surfacic connectivity matrix:
poject the seed mask onto surface.
separate the connectivity matrix and create the profile image (Nifti) for each target, project the value onto surface (mri_vol2surf)
sum all the projection and extract the connectivity profile for the seed region.

"""
import nibabel as nib
import nibabel.gifti as ng
import numpy as np
import os
import os.path as op
import sys
import joblib as jl
import commands


#hemi = 'lh'
#altas = 'destrieux'

hemi = str(sys.argv[1])
tracto_name = str(sys.argv[2])
seed_name = str(sys.argv[3])
projection_method = '--projfrac-avg 0 1 0.1'
#projection_method = '--projfrac 0.5'
fs_exec_dir = '/hpc/soft/freesurfer/freesurfer/bin'
root_dir = '/hpc/crise/hao.c/data'
subject_list = os.listdir(root_dir)

for subject in subject_list[0:1]:

    subject_path = op.join(root_dir,subject)

    tracto_dir = op.join(subject_path, tracto_name)
    surface_dir = op.join(root_dir, subject, 'surface')
    if not op.isdir(surface_dir):
        os.mkdir(surface_dir)

    seedroi_nii_path = op.join(subject_path,'freesurfer_seg', '{}_{}.nii.gz'.format(hemi, seed_name))
    print seedroi_nii_path
    seedroi_gii_path = op.join(subject_path,'freesurfer_seg', '{}_proj_{}.gii'.format(hemi, seed_name))

    coord_file_path = op.join(tracto_dir, 'coords_for_fdt_matrix2')

    connmat_path = op.join(tracto_dir,'conn_matrix_seed2parcels.jl')
    connmat_proj_path = op.join(tracto_dir, 'surfacic_connectivity_profile_STSSTG_{}_proj.jl'.format(hemi.lower()))

    if  not op.isfile(connmat_proj_path):
        print('\ncompute subject : '.format(tracto_dir))
        # ============== project the seed roi volume mask onto surface==========================================================
        if not op.isfile(seedroi_gii_path):
            print 'project the seed roi volume mask onto surface,'
            proj_cmd = '%s/mri_vol2surf --src %s --o %s --out_type gii --regheader %s --hemi %s %s  ' % (fs_exec_dir, seedroi_nii_path, seedroi_gii_path, subject, hemi, projection_method )
            commands.getoutput(proj_cmd)
            print 'seed roi gifti file path : {}'.format(seedroi_gii_path)

        # ============== compute the surfacic connmat============================================================================
        print 'compute surfacic connectivity matrix......\n%s'%connmat_path
        with open(coord_file_path,'r') as f:
            file = f.read()
            text = file.split('\n')
            # print("total number of voxels:", len(text)-1)

            seed_coord = np.zeros((len(text)-1, 3), dtype= int)

            for i in range(len(text)-1):
            #
                data = np.fromstring(text[i], dtype=int, sep=" ")
            #
                seed_coord[i, 0] = data[0]
            #
                seed_coord[i, 1] = data[1]
            #
                seed_coord[i, 2] = data[2]

        mat = jl.load(connmat_path)
        nii = np.zeros((256,256,256))
        n_target = mat.shape[1]

        seedroi_gii = ng.read(seedroi_gii_path)
        seedroi_gii_data = seedroi_gii.darrays[0].data
        sh = seedroi_gii_data.shape
        print ('initial seedroi gifti data shape: {}'.format(sh))

        size = sh[0]
        g_data = np.zeros((size, n_target))

        # compute the profile (nifti image) for each target
        for t in range(n_target):
            valeur = mat[:,t]

        #   compute the nifti image corresponding to this target
            for j in range(len(seed_coord)):
                c = seed_coord[j]
                nii[c[0],c[1],c[2]] = valeur[j]
        #   save the nifti image
            seed_nii = nib.load(seedroi_nii_path)
            img = nib.Nifti1Image(nii, seed_nii.get_affine())
            nii_output_path = op.join(surface_dir, '%03d.nii'%(t+1))
            img.to_filename(nii_output_path)

        #   project each column onto surface
            gii_text_output_path = op.join(surface_dir, '{}.gii'.format(t))
            proj_cmd = '%s/mri_vol2surf --src %s --o %s --out_type gii --regheader %s --hemi %s %s  ' % (fs_exec_dir, nii_output_path, gii_text_output_path, subject, hemi, projection_method )
            commands.getoutput(proj_cmd)
            #print proj_cmd

        #   reload gifti file
            g = ng.read(gii_text_output_path).darrays
            g_data[:,t] = g[0].data

        #   remove gifti and nifti file
            os.remove(nii_output_path)
            os.remove(gii_text_output_path)

        print 'shape of gift data: {} '.format(g_data.shape)

        # extract the profile of the seed region and compute the profile project on the surface
        sum = g_data.sum(axis=1)
        idx = np.where(sum!=0)[0]

        # compare the seef.gii with the projection:
        nb_sumprofil = len(idx)
        nb_vertex_seedroi = len(np.flatnonzero(seedroi_gii_data))
        print ('extract seed region: number of vertex {}'.format(nb_sumprofil))
        print ('number of vertex in seed gifti file:{}'.format(nb_vertex_seedroi))

        print('correct seedroi_gii surface mask')
        seedroi_gii.darrays[0].data[:]=0
        seedroi_gii.darrays[0].data[idx]=1
        ng.write(seedroi_gii, seedroi_gii_path)

        connmat_proj = g_data[idx,:]

        print("shape of the surfacic connmat :{}".format(connmat_proj.shape))

        # save the projected connmat
        jl.dump(connmat_proj, connmat_proj_path, compress=3)
        print "surfacic_connectivity_profile saved: \n %s " %connmat_proj_path

#       check the seedroi gifti file again to make sure the number of vertex correspond to the surfacic matrix
        print('new seed gifti file shape:{}'.format(ng.read(seedroi_gii_path).darrays[0].data.shape))


