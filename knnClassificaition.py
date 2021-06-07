import pandas as pd 
import datetime
import numpy as np
import matplotlib.pyplot as plt

import time
import datetime
import threading
#from mpu6050 import mpu6050

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import tree
import pickle


dir = './'
fileList = ['SensorDataFile']

#mpu = mpu6050(0x68)

def getSensorData(type):
    stack = []
    for f in fileList:
        data = pd.read_csv(dir + f + '.csv')

        #Timestamp 조정 (430ms)
        sampleRate = 430
        data['timestamp'] = data.index * sampleRate

        #해당 값만 뽑아와서 x, y,z로 다시 df 만들기
        if type == 'acc':
            #Ax,Ay,Az
            data = data.drop(['Gx\n(deg/s)','Gy\n(deg/s)','Gz\n(deg/s)'], axis=1)
            data.rename(columns={'Ax\n(m/s^2)':'x', 'Ay\n(m/s^2)':'y', 'Az\n(m/s^2)':'z'}, inplace=True)
        else:
            #Gx,Gy,Gz
            data = data.drop(['Ax\n(m/s^2)','Ay\n(m/s^2)','Az\n(m/s^2)'], axis=1)
            data.rename(columns={'Gx\n(deg/s)':'x', 'Gy\n(deg/s)':'y', 'Gz\n(deg/s)':'z'}, inplace=True)
        
        data['mag'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2)

        data = data.drop(['Date\n(YY:MM:DD)', 'Time\n(HH:MM:SS)', 'Time\n(µs)'], axis=1)

        if len(stack) == 0:
            stack = data
        else:
            stack = pd.concat([data, stack], ignore_index=True)
    return stack


def timedomain_scaled(data):
    WIN_SIZE_IN_MS = 3000
    OVERLAP_RATIO = 0.25
    START_TIME, END_TIME = data['timestamp'].min(), data['timestamp'].max() 

    FEATURES_TIME = []
    WINDOWS = np.arange(START_TIME + WIN_SIZE_IN_MS, END_TIME, WIN_SIZE_IN_MS * (1 - OVERLAP_RATIO))

    for w in WINDOWS:
        win_start, win_end = w - WIN_SIZE_IN_MS, w

        print(w)
        for var in ['x', 'y', 'z', 'mag']:
            # select the rows that belong to the current window, w
            value = data.loc[(win_start <= data['timestamp']) & (data['timestamp'] < win_end), var].values
            #print(value)
            
            # extract basic features 
            min_v = value.min() # min
            max_v = value.max() # max
            mean_v = value.mean() # mean
            std_v = value.std() # std. dev.
            
            # append each result (w: current window's end-timestamp, extracted feature) as a new row
            FEATURES_TIME.append((w, '{}-{}'.format('Min', var), min_v))
            FEATURES_TIME.append((w, '{}-{}'.format('Max', var), max_v))
            FEATURES_TIME.append((w, '{}-{}'.format('Mean', var), mean_v))
            FEATURES_TIME.append((w, '{}-{}'.format('Std', var), std_v))
 
    FEATURES_TIME = pd.DataFrame(FEATURES_TIME, columns=['timestamp', 'feature', 'value'])

    # Reshape data to produce a pivot table based on column values
    FEATURES_TIME = FEATURES_TIME.pivot(index='timestamp', columns='feature', values='value').reset_index()

    #Scaling MinMax
    #scaled = MinMaxScaler().fit_transform(FEATURES_TIME.drop(['timestamp'], axis=1))

    #FEATURES_TIME_SCALED = pd.DataFrame(np.column_stack([FEATURES_TIME['timestamp'], scaled]),columns=FEATURES_TIME.columns)
    
    return FEATURES_TIME


def KMeanCulster(data):
    x = data.drop(columns=['timestamp'])

    model = KMeans(n_clusters=4, algorithm='auto')  #n=target number of shake motion
    model.fit(x)

    return model


def evalModel(model, data):
    x = data.drop(columns=['timestamp'])

    predict = pd.DataFrame(model.predict(x))
    predict.columns=['predict']
    r = pd.concat([x, predict], axis=1)
    #print(r)

    plt.scatter(r['Mean-mag'], r['Max-mag'], c=r['predict'],label=r['predict'], alpha=0.5)
    plt.show()


def main():
    #Acce
    FEATURES_TIME_SCALED = timedomain_scaled(getSensorData('gyro'))
    print(FEATURES_TIME_SCALED)

    #KNN Clustering
    model = KMeanCulster(FEATURES_TIME_SCALED)

    evalModel(model, FEATURES_TIME_SCALED)

    #save model
    pkl_filename = "shake_classification.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    

if __name__ == "__main__":
    main()
