import pandas as pd
import csv

doc = pd.read_csv('VoiceLoc_Database_March16_Virginia.csv')
data = pd.DataFrame(doc)

DTI_scan = data["DTI SCAN [Yes/No]"]
print (DTI_scan)

# DTI_scan = pd.DataFrame(doc, columns=[['DTI SCAN [Yes/No]']=='Y'])

for row in range(0,len(DTI_scan)):
    if DTI_scan(row) == 'Y':
        DTIidx = row

print(DTIidx)
