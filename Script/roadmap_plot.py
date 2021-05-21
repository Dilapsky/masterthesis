# Run in personal computer
import numpy as np
import matplotlib.pyplot as plt
# Run in personal computer
import pandas as pd
from shapely.geometry import MultiLineString
from shapely.wkt import dumps, loads
import time, datetime
import os
import gc

#change the initial_timeStamp according to the unix time of transportation status
initial_timeStamp = 1539273600+600*140
long_lowerbound = 104.0402
lati_lowerbound = 30.6516
long_upperbound = 104.1298
lati_upperbound = 30.7284
Newdf = pd.DataFrame([],columns = ['ID','Longitude','Latitude','LineN'])
#Open network dataset
with open('../Data/Section_201810192021/road_boundary.txt','r',encoding = 'utf-8') as f:
    data = f.readlines()
    for i in range(len(data)-1):
        listdata = data[i+1].strip('\n').split('\t')
        MLS = loads(listdata[2])
        for n,g in enumerate(MLS.geoms):
            for c in g.coords:
                if (c[0]>long_lowerbound) and (c[0]<long_upperbound) and (c[1]<lati_upperbound) and(c[1]>lati_lowerbound):
                    new = pd.DataFrame({'ID':int(listdata[0]),'Longitude':c[0],'Latitude':c[1],'LineN':n},index = [i])
                    Newdf = Newdf.append(new,ignore_index = False)
                else:
                    pass
#Open a part of TTI and Average Speed dataset
tticsv = pd.read_csv('../Data/2018-10-01-00_2018-11-30-00_tti.csv',header = 0,index_col = 'Unnamed: 0')
tticsv.columns = ['ID','datetime','TTI','AVGSpeed']

#processnum is the number of roadmap images that you want to generate for one batch. For personal computer, use a value less than 30 to reduce buffer memory.
processnum = 4
for s in range(processnum):
    print('Now process: ',s)
    if s == 20:
        gc.collect()
        print('gc.collect')
        
    now_timeStamp = initial_timeStamp + 600*s
    timeArray = time.localtime(now_timeStamp)
    otherStyleTime1 = time.strftime("%Y/%m/%d %H:%M:%S",timeArray)
    otherStyleTime2 = time.strftime("%Y-%m-%d %H:%M:%S",timeArray)
    otherStyleTime3 = time.strftime("%Y-%m-%d-%H-%M-%S",timeArray)
    mincsv = tticsv[tticsv['datetime'] == otherStyleTime2]
    TTIarray = np.array(mincsv['TTI'])
    AVGSpeedarray = np.array(mincsv['AVGSpeed'])
    MedianAvgSpeed = np.median(AVGSpeedarray)
    MedianTTIarray = np.median(TTIarray)
    Mergedf = Newdf.merge(mincsv,on = 'ID',how = 'left')
    Mergedf = Mergedf.fillna({'datetime':otherStyleTime1,'TTI':MedianTTIarray,'AVGSpeed':float(round(MedianAvgSpeed))})
    Drawdf = pd.DataFrame([],columns = ['Draworder','Xcoord','Ycoord','Intspd','TTI'])
    Order = 0

    for i in range(len(Mergedf)-1):
        Draworder_item = Order
        Xcoord = int(round(10000*(Mergedf['Longitude'][i]-long_lowerbound)))
        Ycoord = int(round(10000*(Mergedf['Latitude'][i]-lati_lowerbound)))
        Intspd = float(round(Mergedf['AVGSpeed'][i]))
        newtti = Mergedf['TTI'][i]
        draw_n = pd.DataFrame({'Draworder':Order,'Xcoord':Xcoord,'Ycoord':Ycoord,'Intspd':Intspd,'TTI':newtti},index = [i])
        Drawdf = Drawdf.append(draw_n,ignore_index = False)
        if (Mergedf['ID'][i] == Mergedf['ID'][i+1]) and (Mergedf['LineN'][i] == Mergedf['LineN'][i+1]):
            pass

        else:
            Order += 1
    x = [0,0,896,896,0]
    y = [0,768,768,0,0]
    plt.rcParams['figure.dpi'] = 128
    plt.rcParams['savefig.dpi'] = 128
    plt.rcParams['figure.figsize']=(7,6)
    plt.rcParams['image.cmap'] = 'gray'
    plt.figure()
    plt.subplots_adjust(left = -0.05, bottom = -0.05, right = 1.05, top = 1.05, hspace = 0, wspace = 0)
    plt.plot(x,y,color = '#000000', linewidth = 0.3)
    EndNum = Drawdf['Draworder'].iloc[-1]
    for j in range(EndNum+1):
        Segment = Drawdf.loc[Drawdf['Draworder'] == j]
        Xpoint = np.array(Segment['Xcoord'])
        Ypoint = np.array(Segment['Ycoord'])
        Spdpixel = Segment['Intspd'].iloc[0]
        hexn = hex(255-int(Spdpixel))
        hexnum = (hexn[2:4] if len(hexn) == 4 else ('0'+hexn[2:3]))
        Plotcolor = str('#'+hexnum+hexnum+hexnum)
        plt.plot(Xpoint,Ypoint,color = Plotcolor, linewidth = 0.3)
    plt.axis('off')
    savestr =otherStyleTime3 + '.png'
    plt.savefig(os.path.join('../Data/Phase1_result/Roadmap_Pic',savestr))