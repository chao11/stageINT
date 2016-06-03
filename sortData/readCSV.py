import pandas as pd
import os

path_database = '/hpc/banco/VOICELOC_DATABASE'
list_sujet = os.listdir(path_database)

# read the csv file
doc = pd.read_csv('VoiceLoc_Database_March16_Virginia.csv')
DTI = doc[doc['DTI SCAN [Yes/No]'] == 'Y']
print ("subjects have DTI scan : %d" % len(DTI))

fileds = doc.columns

# extract the column of the DTIscan code and check if the code is valid ( with '_L')
DTI_scan_code = doc["DTI_SCAN_CODE"]

Nb_sujet = 0
index = 0

DTIuse_code = pd.DataFrame(columns=('Subject CODE', 'DTI_SCAN_CODE'))

for row in range(0, len(DTI_scan_code)):
    # print row
    code = DTI_scan_code[row]
    # print str

    if code == code and "_L" in code:
        # check if it is not Nan,do:
        # print('get one subject %d' % row)
        Nb_sujet += 1
        DTIuse_code.loc[index] = doc.loc[row, ['Subject CODE', 'DTI_SCAN_CODE']]
        index += 1
        sujet_code = doc.loc[row, "Subject CODE"]
        sujet_code = sujet_code.strip('\'')  # delete singal comma

        print sujet_code
        path_sujet = path_database + '/' + sujet_code
        list_data = os.listdir(path_sujet)

print('nombre de sujet DTIcode valide: %d ' % Nb_sujet)

print (DTIuse_code)

# DTIuse_code.to_excel('/hpc/crise/hao.c/DTIsujet.xlsx')
DTIuse_code.to_csv('subject with valide DTIcode.csv')
# DTIuse_code.to_excel = ('/hpc/crise/hao.c/pycharm/valide_sujet_List.xlsx', 'sheet1')



# check if the subject is complet
# DTIsujet_list = pd.read_csv('subject with valide DTIcode.csv')

DTIsujet_list = pd.read_csv('/hpc/crise/hao.c/VOICELOC_list/LISTE_SujetDTI_complet.csv', )

DTIsujet_list = DTIsujet_list['Diffusion'=='Y' and 'ANAT'=='Y' and 'FIELDMAP']