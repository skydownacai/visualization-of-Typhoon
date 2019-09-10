import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas
import xlrd
import xlwt
import numpy
import math
from keras import regularizers
import traceback
from mpl_toolkits.basemap import Basemap
denglu_index = {}
total_Coast_point = 22#从0开始计算
up_index = 19
passindex = {}
def GET_DATA():
    keyword = ['name_datetime', 'I', 'LAT', 'LON', 'PRES', 'WND', 'OWD', 'END_TAG']
    ALL_DATA = {'name_datetime': [], 'I': [], 'LAT': [], 'LON': [], 'PRES': [], 'WND': [], 'OWD': [], 'END_TAG': []}
    data = xlrd.open_workbook('1981-2018总表.xls').sheet_by_index(0)
    for i in range(len(keyword)):
        if keyword[i] == 'LAT' or keyword[i] == 'LON':
            ALL_DATA[keyword[i]] += [float(item) / 10 for item in data.col_values(i)[1:]]
        else:
            ALL_DATA[keyword[i]] += data.col_values(i)[1:]
    return ALL_DATA
def if_influence_this_province_one_point(shapefile,lat,lon):
    left_up = False
    left_down = False
    right_up = False
    right_down = False
    for Boundary in shapefile:
        if Boundary[0] > lon and Boundary[1] > lat:
            right_up = True
        if Boundary[0] > lon and Boundary[1] < lat:
            right_down = True
        if Boundary[0] < lon and Boundary[1] > lat:
            left_up = True
        if Boundary[0] < lon and Boundary[1] < lat:
            left_down = True
        if  left_up and left_down and right_up and right_down:
            return True
    return False
def if_influence_this_province_path(shapefile,path,pathid,province):
    for i in range(len(path)):
        point = path[i]
        if if_influence_this_province_one_point(shapefile,point[0],point[1]):
            denglu_index[pathid] = [i,point,province]
            return True
    denglu_index[pathid] = 0
    return  False
def if_influence(SHAPE_FILE,path,pathid):
        for province in SHAPE_FILE:
            if if_influence_this_province_path(SHAPE_FILE[province],path,pathid,province):
                return True
        return False
def path_data():
    ALL_path = {}
    data = GET_DATA()
    path_id = []
    for i in range(len(data['name_datetime'])):
        ID =  data['name_datetime'][i][:9]
        if ID not in ALL_path:
            ALL_path[ID] = []
            path_id .append(ID)
        TIM = data['name_datetime'][i][5:]
        lat = data['LAT'][i]
        lon = data['LON'][i]
        PRE = data['PRES'][i]
        WND = data['WND'][i]
        END = data['END_TAG'][i]
        LEL = data['I'][i]
        ALL_path[ID].append((lat,lon,PRE,WND,END,TIM,LEL))
    return (ALL_path,path_id)
def load_4shapfile(filepath):
    map = Basemap(projection='cyl', lat_0=90, lon_0=160, \
                  llcrnrlat=0, urcrnrlat=45, \
                  llcrnrlon=90, urcrnrlon=180, \
                  rsphere=6371200., resolution='h', area_thresh=1000)

    map.readshapefile(filepath, 'states', drawbounds=False, color='black', linewidth=0.4)
    SHAPE_FILE = {}
    for info, shp in zip(map.states_info, map.states):
        province = info['NAME_1']  # name of state
        if province not in ['Zhejiang','Shanghai','Guangdong','Fujian']:
            continue
        if province not in SHAPE_FILE:
            SHAPE_FILE[province] = [shp]
        else:
            SHAPE_FILE[province].append(shp)
    for key in SHAPE_FILE:
        b = list(sorted(SHAPE_FILE[key], key=lambda e: len(e), reverse=True))
        SHAPE_FILE[key] = b[0]
    print('读取4省shapefile成功')
    return SHAPE_FILE
def wrtie_influence():
    a  = pandas.read_csv('influence.csv')
    return (list(a['ID'].values),list(a['influence'].values))
def piecelinear_of_coastline():
    data = {}
    coastline = coast_line()
    for i in range(total_Coast_point):
        x1 = coastline[i]
        x2 = coastline[i+1]
        data [i] = {
            'coef':(x1[1] - x2[1]) / (x1[0] - x2[0]),
            'start_y':x1[1],
            'start_x':x1[0],
            'end_x'  :x2[0],
            'end_y'  :x2[1]
        }
    return data
def if_point_in_coastline(coast_line_piecelinear,point):
    '判断一个点是否在海岸线内'
    '注意高精度需求'
    for i in range(up_index + 1):
        point_x = point[1]
        point_y = point[0]
        start_x = coast_line_piecelinear[i]['start_x']
        start_y = coast_line_piecelinear[i]['start_y']
        end_x   = coast_line_piecelinear[i]['end_x']
        end_y   = coast_line_piecelinear[i]['end_y']
        if point_x >= start_x and point_x < end_x and point_y <= 32:
            if point_y*(end_x - start_x) >= (end_y - start_y)*(point_x - start_x) + start_y * (end_x - start_x):
                if point_x > 121.972356:
                    if point_x <= 122.037456:
                        if  point_y*(122.037456 - 121.972356) <= (30.633228  - 31.591419)*(point_x - 121.972356) + 31.591419*(122.037456 - 121.972356):
                            return i
                    elif point_x <= 122.098542:
                        if  point_y*(122.098542 - 122.037456) <= (30.099967  - 30.633228)*(point_x - 122.037456) + 30.633228*(122.098542 - 122.037456):
                            return i
                else:
                    return  i
def get_denglu_index():
    '返回每个台风路径第几个点已经登录 与登录前的一个点,与在海岸线的哪个片段'
    'point[0] = lat'
    'point[1] = lon'
    pathdata = path_data()[0]
    coast_line_piecelinear = piecelinear_of_coastline()
    data = {}
    ALL = {'ID':[],'denglu':[]}
    count = 0
    for pathid in pathdata:
        path = pathdata[pathid]
        denglu_index = -1
        data[pathid] = denglu_index
        ALL['ID'].append(pathid)
        find = False
        for i in range(len(path)):
            point = path[i]
            RETURN = if_point_in_coastline(coast_line_piecelinear,point)
            if RETURN != None :
                count += 1
                data[pathid] = [point,path[i-1],RETURN]
                find= True
                break
        if find:
            ALL['denglu'].append(1)
        else:
            ALL['denglu'].append(0)
    print('登录台风个数为:',count)
    pandas.DataFrame(ALL).to_csv('登录台风统计.csv')
    return data
def path_vector_knn():
    data = {
        'pathid':[],
        'start_x':[],
        'start_y':[],
        'destination_x':[],
        'destination_y': [],
        'angle':[]
    }
    pathdata = path_data()[0]
    for pathid in pathdata:
        s_x = pathdata[pathid][0][1]
        s_y = pathdata[pathid][0][0]
        e_x = pathdata[pathid][-1][1]
        e_y = pathdata[pathid][-1][0]
        data['pathid'] .append(pathid)
        data['start_x'].append(s_x)
        data['start_y'].append(s_y)
        data['destination_x'].append(e_x)
        data['destination_y'].append(e_y)
        data['angle'].append((s_x*e_x + s_y*e_y)/(math.sqrt(s_x*s_x+s_y*s_y) * math.sqrt(e_x*e_x+e_y*e_y)))
    pandas.DataFrame(data).to_csv('KNN_path_vector.csv')
def str_to_latlon(x):
    x = x.replace("\\",'').replace('\n','').replace('"','')
    du_index = x.index('°')
    fen_index = x.index("'")
    du =  int(x[:du_index])
    fen = int(x[du_index + 1: fen_index])
    sec = float(x[fen_index +1 :])
    return round(du + fen/60 + sec/3600,6)
def coast_line():
    global total_Coast_point
    '获得海岸线分割点数据'
    with open('coastline.txt') as f:
         raw = f.readlines()
    coastline = {}
    total_Coast_point = -1
    for this_item in raw:
        total_Coast_point += 1
        item = this_item.split(' ')
        coastline[total_Coast_point] = (str_to_latlon(item[2]),str_to_latlon(item[1]))
    return coastline
def line_of_two_point(x1, x2):
    coef = (x1[1] - x2[1]) / (x1[0] - x2[0])
    x = numpy.linspace(x1[0], x2[0], 1000)
    return x,numpy.array(list([coef*(x0 - x1[0]) + x1[1] for x0 in x]))
def cross_point(line2, line1):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]
    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]
    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]
def get_denglu_point():
    '得到所有台风在海岸线上的登陆点'
    denglu = get_denglu_index()
    coastline = piecelinear_of_coastline()
    ALL = []
    for pathid in denglu:
        if denglu[pathid] == -1:
            passindex[pathid] = -1
            continue
        denglu_point = denglu[pathid][0]
        last_point = denglu[pathid][1]
        find = False
        for piece_index in range(0, total_Coast_point):
            piece = coastline[piece_index]
            crosspoint = cross_point([denglu_point[1], denglu_point[0], last_point[1], last_point[0]],
                                     [piece['start_x'], piece['start_y'], piece['end_x'], piece['end_y']])
            if piece['start_x'] < piece['end_x']:
                if crosspoint[0] >= piece['start_x'] and crosspoint[0] <= piece['end_x'] and crosspoint[0] >= min(
                        denglu_point[1], last_point[1]) and crosspoint[0] <= max(denglu_point[1], last_point[1]):
                    d_l = (crosspoint[0] - last_point[1]) ** 2 + (crosspoint[1] - last_point[0]) ** 2
                    d_d = (crosspoint[0] - denglu_point[1]) ** 2 + (crosspoint[1] - denglu_point[0]) ** 2
                    l_l = int(last_point[-1])
                    l_d = int(denglu_point[-1])
                    l_c = (d_l * l_l + d_d * l_d) / (d_l + d_d)
                    # 用加权平均计算登陆点台风的等级
                    crosspoint.append(l_c)
                    ALL.append(crosspoint)
                    passindex[pathid] = piece_index
                    find = True
            else:
                if crosspoint[0] <= piece['start_x'] and crosspoint[0] >= piece['end_x'] and crosspoint[0] >= min(
                        denglu_point[1], last_point[1]) and crosspoint[0] <= max(denglu_point[1], last_point[1]):
                    d_l = (crosspoint[0] - last_point[1]) ** 2 + (crosspoint[1] - last_point[0]) ** 2
                    d_d = (crosspoint[0] - denglu_point[1]) ** 2 + (crosspoint[1] - denglu_point[0]) ** 2
                    l_l = int(last_point[-1])
                    l_d = int(denglu_point[-1])
                    l_c = (d_l * l_l + d_d * l_d) / (d_l + d_d)
                    # 用加权平均计算登陆点台风的等级
                    crosspoint.append(l_c)
                    ALL.append(crosspoint)
                    passindex[pathid] = piece_index
                    find = True
    return ALL
def get_denglu_point2():
    '得到所有台风在海岸线上的登陆点'
    denglu = get_denglu_index()
    coastline = piecelinear_of_coastline()
    ALL = {}
    for pathid in denglu:
        if denglu[pathid] == -1:
            passindex[pathid] = -1
            continue
        denglu_point = denglu[pathid][0]
        last_point = denglu[pathid][1]
        find = False
        for piece_index in range(0, total_Coast_point):
            piece = coastline[piece_index]
            crosspoint = cross_point([denglu_point[1], denglu_point[0], last_point[1], last_point[0]],
                                     [piece['start_x'], piece['start_y'], piece['end_x'], piece['end_y']])
            if piece['start_x'] < piece['end_x']:
                if crosspoint[0] >= piece['start_x'] and crosspoint[0] <= piece['end_x'] and crosspoint[0] >= min(
                        denglu_point[1], last_point[1]) and crosspoint[0] <= max(denglu_point[1], last_point[1]):
                    d_l = (crosspoint[0] - last_point[1]) ** 2 + (crosspoint[1] - last_point[0]) ** 2
                    d_d = (crosspoint[0] - denglu_point[1]) ** 2 + (crosspoint[1] - denglu_point[0]) ** 2
                    l_l = int(last_point[-1])
                    l_d = int(denglu_point[-1])
                    l_c = (d_l * l_l + d_d * l_d) / (d_l + d_d)
                    # 用加权平均计算登陆点台风的等级
                    crosspoint.append(l_c)
                    passindex[pathid] = piece_index
                    find = True
                    ALL[pathid] = crosspoint
            else:
                if crosspoint[0] <= piece['start_x'] and crosspoint[0] >= piece['end_x'] and crosspoint[0] >= min(
                        denglu_point[1], last_point[1]) and crosspoint[0] <= max(denglu_point[1], last_point[1]):
                    d_l = (crosspoint[0] - last_point[1]) ** 2 + (crosspoint[1] - last_point[0]) ** 2
                    d_d = (crosspoint[0] - denglu_point[1]) ** 2 + (crosspoint[1] - denglu_point[0]) ** 2
                    l_l = int(last_point[-1])
                    l_d = int(denglu_point[-1])
                    l_c = (d_l * l_l + d_d * l_d) / (d_l + d_d)
                    # 用加权平均计算登陆点台风的等级
                    crosspoint.append(l_c)
                    ALL[pathid] = crosspoint
                    passindex[pathid] = piece_index
                    find = True
        if not find:
            ALL[pathid] = -2
            print(pathid)
    return ALL
def get_denglu_pathid():
    denglus = get_denglu_index()
    all = []
    for id in denglus:
        if denglus[id] != -1:
            all.append(id)
    return all
def get_type_of_denlgu():
    return  pandas.read_csv('sum_denglu_cluster.csv',encoding='gbk')['类别'].values
def get_denglu_count():
    denglu_id =  get_denglu_pathid()
    pathes = path_data()[0]
    ALL = {}
    count = 0
    for id in pathes:
        if id not in denglu_id:
            continue
        count += 1
        point = pathes[id][0]
        times = point[-2][:6]
        if times not in ALL:
            ALL[times] = 1
        else:
            ALL[times] += 1
    DATA  = {'时间':[],'登录数':[]}
    for year in range(1981,2019):
        for month in range(1,13):
            if month < 10:
                this_time = '{}0{}'.format(year,month)
            else:
                this_time = '{}{}'.format(year, month)
            DATA['时间'].append(this_time)
            if this_time not in ALL:
                DATA['登录数'].append(0)
            else:
                DATA['登录数'].append(ALL[this_time])
    pandas.DataFrame(DATA).to_csv('登录次数年月统计.csv')
def get_denglu_province():
    get_denglu_point()
    ALL = {'ID':[],'province':[]}
    data = {}
    for id in passindex:
        ALL['ID'].append(id)
        if passindex[id] == -1:
            ALL['province'].append('未登录')
            data[id]='未登录'
        elif passindex[id] <= 11:
            ALL['province'].append('广东')
            data[id] = '广东'
        elif passindex[id] <= 14:
            ALL['province'].append('福建')
            data[id] = '福建'
        elif passindex[id] <= 20:
            ALL['province'].append('浙江')
            data[id] = '浙江'
        else:
            ALL['province'].append('上海')
            data[id] = '上海'
    print(data)
    pandas.DataFrame(ALL).to_csv('台风登录省份统计.csv')
    return data
def get_province_count():
    data = pandas.read_csv('denglu.csv',encoding='gbk')
    y = []
    raw_y = data['cishu'].values
    for item in raw_y:
        y.append(item)
    return y
def dataset(cut):
    seq = get_province_count()
    x = []
    y = []
    for i in range(cut,len(seq)):
        y.append(seq[i])
        x.append([item[0]for item in seq[i-cut:i]])
    return (x,y)
def detali_denglu_point():
    denglupoints = get_denglu_point2()
    data = get_denglu_index()
    count = 0
    book = xlwt.Workbook()
    sheet = book.add_sheet('0')
    for id in data:
        detail = data[id]
        if detail == -1:
            continue
        denglupoint = denglupoints[id]
        sheet.write(count, 0,id)
        #format of point:(lat,lon,PRE,WND,END,TIM,LEL)
        if denglupoint != -2:
            sheet.write(count, 1, denglupoint[0])
            sheet.write(count, 2, denglupoint[1])
        else:
            sheet.write(count, 1, '无')
            sheet.write(count, 2, '无')
        afterpoint = detail[0]
        beforepoint = detail[1]
        sheet.write(count,3,beforepoint[1])
        sheet.write(count,4,beforepoint[0])
        sheet.write(count,5,int(beforepoint[-1]))
        sheet.write(count,6,int(beforepoint[3]))
        sheet.write(count,7,int(beforepoint[2]))
        sheet.write(count,8,afterpoint[1])
        sheet.write(count,9,afterpoint[0])
        sheet.write(count,10,int(afterpoint[-1]))
        sheet.write(count,11,int(afterpoint[3]))
        sheet.write(count,12,int(afterpoint[2]))
        count += 1
    book.save('outcome.xls')
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
def ROUND(Y):
    y = Y
    for i in range(len(y)):
        y[i][0] = round(y[i][0])
        if y[i][0] < 0:
            y[i][0] = 0
    return y
dataset = np.array(get_province_count())
print(dataset)
dataset = dataset.astype('float32')  # confirm the type as 'float32'
dataset = dataset.reshape(-1, 1)
# fix random seed for reproducibility
numpy.random.seed(7)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

future_num = 12
def LSTM_MODEL_AND_FIT(epoch,look_back,unit,Lambda):
    # use this function to prepare the train and test datasets for modeling
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    #x like : [
    #  [[x1]]
    #  [[x2]
    #  ...
    #  [[x3]
    # ]

    #y like : [
    #  [y1]
    #  [y2]
    #  ...
    #  [yn]
    # ]
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(unit, input_shape=(1, look_back),kernel_regularizer = regularizers.l2(Lambda)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epoch, batch_size=10, verbose=0)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)


    future_x  = testX[-1:]
    future_x[0] = np.append(future_x[0][0][1:],testPredict[-1][0])
    future_y = []
    for i in range(future_num):
         future_y .append(list(model.predict(future_x)[0]))
         future_x[0] = np.append(future_x[0][0][1:], future_y[0][0])

    future_y = np.array(future_y).reshape(-1,1)
    future_y = scaler.inverse_transform(future_y)
    future_y = np.array(list(future_y)[:6]+list(future_y)[8:]+list(future_y)[6:8]).reshape(-1,1)
    print(future_y)
    future_plot = np.empty_like(dataset)
    future_plot[:,:] = np.nan
    future_plot = np.append(future_plot,future_y)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    #trainPredict = ROUND(trainPredict)

    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    #testPredict = ROUND(testPredict)

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    # plot baseline and predictions
    figure = plt.figure(figsize=(16,8))
    plt.plot(scaler.inverse_transform(dataset),c='blue')
    plt.plot(trainPredictPlot,c='red',label = '训练集预测')
    plt.plot(testPredictPlot,c ='green',label = '测试集预测')
    plt.plot(future_plot, c='purple',label = '2019年登陆台风次数预测'.format(future_num))
    #print(testPredictPlot[-12:])
    #plt.savefig('{}_{}_{}.png'.format(look_back,unit,epoch))
    plt.legend(loc = 'upper right',prop = {'size':18})
    #plt.title('unit:{} lookback:{} epoch:{} lambada:{} test_s:{}  train_s:{}'.format(unit,look_back,epoch,Lambda,testScore,trainScore))
    plt.show()
    print(future_y)
    print(look_back,';',unit,';',epoch,';',trainScore,';',testScore)
# create and fit the LSTM network
LSTM_MODEL_AND_FIT(600,50,90,0.00004)