import pandas as pd


# read the csv file
doc = pd.read_csv('VoiceLoc_Database_March16_Virginia.csv')
DTI = doc[doc['DTI SCAN [Yes/No]']=='Y']
print ("subjects have DTI scan : %d" % len(DTI))


fileds = doc.columns

# extract the colonne of the DTIscan code and chack if the code is valided with '_L'
DTI_scan_code = doc["DTI_SCAN_CODE"]

Nb_sujet = 0
index = 0


DTIuse_code = pd.DataFrame(columns=('Subject CODE', 'DTI_SCAN_CODE'))

for row in range(0, len(DTI_scan_code)):
    # print row
    code = DTI_scan_code[row]
    # print str

    if code == code:  # check if it is not Nan,do:
        if "_L" in code:
            # print('get one subject %d' % row)
            Nb_sujet += 1
            DTIuse_code.loc[index] = doc.loc[row, ['Subject CODE', 'DTI_SCAN_CODE']]
            index += 1


print('nombre de sujet DTIcode valide: %d ' % Nb_sujet)

print (DTIuse_code)

# DTIuse_code.to_excel('/hpc/crise/hao.c/DTIsujet.xlsx')
DTIuse_code.to_csv('subject with valide DTIcode.csv')
DTIuse_code.to_excel = ('D:\stage\stageINT-master\stageINT-master\pycharm\valide_sujet_List.xlsx', 'sheet1')

pd.read_csv('subject with valide DTIcode.csv')