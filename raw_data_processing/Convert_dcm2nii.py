# convert the DICOM data to NII format
import os.path as op
import os
import commands
import sujet_profiles as sp


# create directory for the subject in the workspace
def checkDir(path):
    # os.listdir(path)
    if not op.exists(path):
        os.makedirs(path)
        complete = 0
        print ('create the empty directory for raw_anat in workspace')
    elif len(os.listdir(path)) < 3:
        complete = 0
        print ('The directory exists but not complete.')
    else:
        complete = 1
        print (os.listdir(path))
        print('convert complete')
    return complete


def movefiles(source_path, dir_path):
    os.chdir(source_path)
    cmd_mvfiles = 'mv *.nii.gz *.bvec *.bval {}'.format(dir_path)
    print (cmd_mvfiles)
    commands.getoutput(cmd_mvfiles)
    return


def dcm2nii(source,workspace):
    print("convert the data from DICOM to Nii...")
    cmd = 'dcm2nii {} '.format(source)
    print("$ "+cmd)
    commands.getoutput(cmd)

    movefiles(source,workspace)
    return


def convert(subjectCode, DTIcode):
    subject = sp.Subject(subjectCode)
#    print subject.anat_source, subject.anat_workspace
    # check if the datas are already converted, if not, convert the data and move to the workspace
    # convert the anat data and move to the workspace

    if op.exists(subject.anat_source):
        complete = checkDir(subject.anat_workspace)

        if complete == 0:
           dcm2nii(subject.anat_source,subject.anat_workspace)
           print(os.listdir(subject.anat_workspace))

    else:
        print("not exist: " + subject.anat_source)

    # convert DWI data and move to the workspace
    if op.exists(subject.dwi_source):
        complete = checkDir(subject.dwi_workspace)

        if complete == 0:
            dcm2nii(subject.dwi_source, subject.dwi_workspace)
            print(os.listdir(subject.dwi_workspace))
    else:
        print("not exist: "+subject.dwi_source)

    return
