import os

class Subject:
    """source path and workspace path for the dwi and ANAT data"""

    def __init__(self,subject):
        self.subject = subject

        list = os.listdir("/hpc/banco/VOICELOC_DATABASE/{}/DTI/".format(self.subject))
        self.DTIcode = list[0]

        path_database = "/hpc/banco/VOICELOC_DATABASE"

        path_subject = '{}/{}/DTI/{}'.format(path_database, self.subject, self.DTIcode)
        self.anat_source = '{}/ANAT/'.format(path_subject)
        self.dwi_source = '{}/DIFFUSION/'.format(path_subject)

        # workspace path
        self.anat_workspace = '/hpc/crise/hao.c/data/{}/raw_anat/'.format(self.subject)
        self.dwi_workspace = '/hpc/crise/hao.c/data/{}/raw_dwi/'.format(self.subject)


## test for the class


