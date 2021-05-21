# Run in Kaggle Kernel
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#increase the number from 0 to the total number of files you have in the source. 
nowfile = filenames[64]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Nrow = len(pd.read_csv(os.path.join(dirname, nowfile),skiprows = 0,header = 0,usecols=[0,1,2,3],index_col = 0) )
raw_data=pd.read_csv(os.path.join(dirname, nowfile),skiprows = 0,header = 0,usecols=[0,1,2,3],index_col = 0,iterator = True)      
long_lowerbound = 104.0402
lati_lowerbound = 30.6516
long_upperbound = 104.1298
lati_upperbound = 30.7284
x = [0,0,896,896,0]
y = [0,768,768,0,0]
plt.rcParams['figure.dpi'] = 128
plt.rcParams['savefig.dpi'] = 128
plt.rcParams['figure.figsize']=(7,6)
plt.rcParams['image.cmap'] = 'gray'
plt.figure()
plt.subplots_adjust(left = -0.05, bottom = -0.05, right = 1.05, top = 1.05, hspace = 0, wspace = 0)
plt.plot(x,y,color = '#000000', linewidth = 0.3)

for i in range(Nrow):
    print('Now process: ',i,' in /',Nrow)
    speedmatrix = []
    cd = raw_data.get_chunk(1)
    Splitdf = pd.DataFrame([],columns = ['OriginalID','Longitude','Latitude','Timestamp'])
    IDList = cd['Trajectory'].iloc[:].index.tolist()
    for n, ele in enumerate(cd['Trajectory']):
        element = ele.strip('[]').split(',')
        for p in range(len(element)):
            elemen = element[p].split(' ')
            elem = [element[p] for element[p] in elemen if element[p]]
            Longdata = elem[0]
            Latidata = elem[1]
            Timedata = elem[2]
            IDdata = IDList[n]
            Split_n = pd.DataFrame({'OriginalID':IDdata,'Longitude':Longdata,'Latitude':Latidata,'Timestamp':Timedata},index = [p])
            Splitdf = Splitdf.append(Split_n,ignore_index = False)
    for k in range(len(Splitdf)-1):
        long0 = float(Splitdf['Longitude'].iloc[k+0])
        lati0 = float(Splitdf['Latitude'].iloc[k+0])
        long1 = float(Splitdf['Longitude'].iloc[k+1])
        lati1 = float(Splitdf['Latitude'].iloc[k+1])
        time0 = float(Splitdf['Timestamp'].iloc[k+0])
        time1 = float(Splitdf['Timestamp'].iloc[k+1])
        if (long0 > long_lowerbound) and (long0 < long_upperbound) and (lati0 > lati_lowerbound) and (lati0 < lati_upperbound) and
        (time0 > int(nowfile[0:10])) and (time0 < int(nowfile[11:21])):
            Xcoord0 = round(10000*(long0-long_lowerbound))
            Xcoord1 = round(10000*(long1-long_lowerbound))
            Ycoord0 = round(10000*(lati0-lati_lowerbound))
            Ycoord1 = round(10000*(lati1-lati_lowerbound))
            Xpoint = [Xcoord0,Xcoord1]
            Ypoint = [Ycoord0,Ycoord1]
            if ((long1-long0)**2+(lati1-lati0)**2) == 0:
                speed1 = 0
            else:
                speed1 = ((long1-long0)**2+(lati1-lati0)**2)**0.5/(time1-time0)*111.19*3600
                speed1 = round(speed1)    
                if (int(speed1) > 150):
                    speed1 = 150.0
                else:
                    pass
            hexn = hex(255-int(speed1))
            hexnum = (hexn[2:4] if len(hexn) == 4 else ('0'+hexn[2:3]))
            if hexnum == '0x':
                print('Problem: ', long1,long0,lati1,lati0,time1,time0,speed1,i,hexn)
                break
            Plotcolor = str('#'+hexnum+hexnum+hexnum)
            plt.plot(Xpoint,Ypoint,color = Plotcolor, linewidth = 0.03)
        else:
            pass
plt.axis('off')
savename = nowfile[0:-4]+'.png'
