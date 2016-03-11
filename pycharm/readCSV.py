import pandas as pd


# read the csv file
doc = pd.read_csv('VoiceLoc_Database_March16_Virginia.csv')
# name of the fields
fileds = doc.columns

# extract the colonne of the DTIscan code and chack if the code is valided with '_L'
DTI_scan_code = doc["DTI_SCAN_CODE"]

Nb_sujet = 0
#DTIuse_code = []

DTIuse_code = pd.DataFrame(columns=('Subject CODE', 'DTI_SCAN_CODE'))

for row in range(0, len(DTI_scan_code)):
    # print row
    code = DTI_scan_code[row]
    # print str

    if code == code:  # check if it is not Nan,do:
        if "_L" in code:
            # print('get one subject %d' % row)
            Nb_sujet += 1
           # DTIuse_code.append(doc.loc[row, ['Subject CODE', 'DTI_SCAN_CODE']])
            DTIuse_code.append(doc.loc[row, ['Subject CODE', 'DTI_SCAN_CODE']])
            # sujetValid.writerow()



print('nombre de sujet DTIcode valide: %d ' % Nb_sujet)

print DTIuse_code

DTIuse_code.to_excel('/hpc/crise/hao.c/DTIsujet.xlsx')

DTIuse_code.to_excel = ('VoiceLoc_Database_March16_Virginia.xlsx', 'sheet1')
