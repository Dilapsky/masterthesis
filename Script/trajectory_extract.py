##Run in personal computer
#Extract 12 hour data for experiment, the whole data is about 2.8 million rows
import pandas as pd
Oct10_Oct20_data=pd.read_csv('../Data/chengdushi_1010_1020.zip',skiprows = 0,
                             header = None,usecols=[0,1,2],compression = 'zip',iterator = True) 
loop = True
chunkSize = 10000
chunks = []
nowdata = 1539273900

#change here
f = open('./trajectory_csv/2018-10-12-00-05_2018-10-12-12-05.csv','w+')
f.write('Driverid_1,Driverid_2,Trajectory'+'\n')
while loop:
    try:
        Oct1020_df = pd.DataFrame(Oct10_Oct20_data.get_chunk(chunkSize))
        for n, ele in enumerate(Oct1020_df[2]):
            element = ele.strip('[]').split(',')
            for p in range(len(element)):
                elem = element[p].split(' ')

                                         #change here
                if int(elem[-1]) > nowdata and int(elem[-1]) < (nowdata+43200):
            #1539950400:2018-10-19 20:00:00; 1539954000:2018-10-19 21:00:00
            #1539957600:2018-10-19 22:00:00; 1539954000:2018-10-19 21:00:00
            # '2018-10-10 00:05:00' 1539101100; '2018-10-10 06:05:00' 1539122700 (+21600);'2018-10-10 12:05:00' 1539144300 1539165900
                   # print(Oct1020_df.iloc[n])
                 #   chunks.append(Oct1020_df.iloc[n])
                    f.write(str(Oct1020_df.iloc[n][0])+','+str(Oct1020_df.iloc[n][1])+','+'\"'+str(Oct1020_df.iloc[n][2])+'\"'+'\n')
                    break
                else:
                    pass

    except StopIteration:
        loop = False
        print("Iteration is stopped.")
print("begin concat")
f.close()
#cd = pd.DataFrame(chunks)
#cd.to_csv('2018-10-10-00-05_2018-10-10-04-05.csv')
print('saved')