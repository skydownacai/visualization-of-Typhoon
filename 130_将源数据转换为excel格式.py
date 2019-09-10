import pandas
import numpy
import xlwt
keyword = ['name_datetime','I','LAT','LONG','PRES','WND','OWD','END_TAG']
def write(DATA,name):
    data = xlwt.Workbook()
    sheet = data.add_sheet('0')
    for i in range(8):

        sheet.write(0,i,keyword[i])
    for j in range(len(DATA[keyword[0]])):
       for t in range(8):
            sheet.write(j+1,t,DATA[keyword[t]][j])
    data.save(name)
def CELAN(DATA):
    return [clean(item.split(' ')) for item in DATA]
def clean(x):
    y = []
    for item in x:
        if item != '':
            y.append(item.replace('\n',''))
    return y
def get_year(DATA):
    datas = CELAN(DATA)
    y = []
    ID = ''
    END_TAG = 0
    for item in datas:
        if len(item) == 9:
            '表示是个头'
            ID = item[3]
            END_TAG = item[-4]
        else:
            item[0] = ID+'_'+item[0]
            if len(item) == 6:
                item.append(' ')
            else:
                print(item[0])
            item.append(END_TAG)
            y.append(item)
    return y
keyword = ['name_datetime','I','LAT','LONG','PRES','WND','OWD','END_TAG']
#ALL_DATA = {'name_datetime':[],'I':[],'LAT':[],'LONG':[],'PRES':[],'WND':[],'OWD':[],'END_TAG':[]}
for i in range(1981,2019):
    ALL_DATA = {'name_datetime': [], 'I': [], 'LAT': [], 'LONG': [], 'PRES': [], 'WND': [], 'OWD': [], 'END_TAG': []}
    with open('CH{}BST.txt'.format(i)) as f:
        data = f.readlines()
        DATA = get_year(data)
        for item in DATA:
            for k in range(8):
                ALL_DATA[keyword[k]].append(item[k])
        #write(ALL_DATA,''.format(i))
        DATAFRAME = pandas.DataFrame(ALL_DATA)
        DATAFRAME.to_csv('year\{}.csv'.format(i),encoding='UTF-8')