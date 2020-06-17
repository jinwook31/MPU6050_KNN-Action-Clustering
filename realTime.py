import pandas as pd
import datetime
import numpy as np

import time
from time import sleep
import datetime
import threading
from mpu6050 import mpu6050

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import tree
import pickle

# Call MPU6050 class
mpu = mpu6050(0x68)

def getRealTimeData(index):
    #MPU6050 functions with mag (Gyro)
    column_titles = ['Date\n(YY:MM:DD)', 'Time\n(HH:MM:SS)', 'Time\n(µs)','Gx', 'Gy', 'Gz']
    current_time = datetime.datetime.now()
    #accel_data = mpu.get_accel_data()
    gyro_data = mpu.get_gyro_data()
    #current_acc = [accel_data['x'], accel_data['y'], accel_data['z']]
    current_gyro = [gyro_data['x'], gyro_data['y'], gyro_data['z']]
    row = [current_time.strftime('%Y-%m-%d'), current_time.strftime('%H:%M:%S'), current_time.strftime('%f')]
    row = row + current_gyro

    data = pd.DataFrame([row], columns=column_titles)

    #Timestamp fixation (430ms)
    sampleRate = 427
    data['timestamp'] = index * sampleRate
    index += 1

    #Gx,Gy,Gz
    #data = data.drop(['Ax','Ay','Az'], axis=1)
    data.rename(columns={'Gx':'x', 'Gy':'y', 'Gz':'z'}, inplace=True)

    data['mag'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2)
    #print(data)

    data = data.drop(['Date\n(YY:MM:DD)', 'Time\n(HH:MM:SS)', 'Time\n(µs)'], axis=1)

    return data


def timedomain_scaled(data, idx):
    FEATURES_TIME = []
    WIN_SIZE_IN_MS = 3000
    w = 3000
    win_start, win_end = data.iloc[0,3], data.iloc[6,3]
    
    for var in ['x', 'y', 'z', 'mag']:
        # select the rows that belong to the current window, w
        value = data.loc[(win_start <= data['timestamp']) & (data['timestamp'] < win_end), var].values  #SOMETHING IS WRONG HERE
        
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

    return FEATURES_TIME


def main():
    index = 0

    # Load from file
    pkl_filename = 'shake_classification.pkl'
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    data = pd.DataFrame()
    while True:
        data = data.append(getRealTimeData(index))
        sleep(0.1)
        index += 1

        if len(data) > 6:
            data = data.reset_index(drop=True)
            #print(data)
            window = timedomain_scaled(data, index)
            #print(window)
            
            predict = pickle_model.predict(window.drop(columns=['timestamp']))
            print(predict[0])  #0: , 1: hold, 2:
            
            #Moving window
            data.drop(data.head(2).index, inplace=True)


if __name__ == "__main__":
    main()
