import pandas as pd
import csv
import numpy as np

# read the csv file
doc = pd.read_csv('VoiceLoc_Database_March16_Virginia.csv')
# name of the fields
fileds = doc.columns

# extract the colonne of the DTIscan code and chack if the code is valided with '_L'
DTI_scan_code = doc["DTI_SCAN_CODE"]

Nb_sujet = 0
DTIuse_code = []
for row in range(0, len(DTI_scan_code)):
    # print row
    str = DTI_scan_code[row]
    # print str

    if str == str:  # check if it is not Nan,do:
        if "_L" in str:
            # print('get one subject %d' % row)
            Nb_sujet += 1
            DTIuse_code.append(doc.loc[row, ['Subject CODE', 'DTI_SCAN_CODE']])
            # sujetValid.writerow()

# DTIuse_code.to_csv = ('VoiceLoc_Database_March16_Virginia.csv')


print('nombre de sujet DTIcode valide: %d' % Nb_sujet)

print DTIuse_code
"""


print(DTI_scan)


with open('VoiceLoc_Database_March16_Virginia.csv','rb')as csvfile:
    write = csv.writer(csvfile) # write into csv

    firstrow = csvfile.readlines(1)
    filednames = tuple(firstrow[0].strip('\n').split('\t')) # get the fieldname
    print (filednames)

    for row in DTI_scan:
        if DTI_scan[row] == 'Y':
            write.writerow = csvfile(row)

    print(write)

    print (DTI_scan)

# DTI_scan = pd.DataFrame(doc, columns=[['DTI SCAN [Yes/No]']=='Y'])
"""
"""
for row in range(0,len(DTI_scan)):
    if DTI_scan(row) == 'Y':
        DTIidx = row

print(DTIidx)
"""
