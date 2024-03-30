import os
import shutil
from datetime import datetime
import time
import csv
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from filter_utils import CustomSubset_two
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Getnslkdd import data_set
classes = [
    "Dos",
    "Normal",
    "Probe",
    "R2L",
    "U2R",
]

model_path = "/home/test/Desktop/for_test/pcap_filter/model0.pth"
work_log = '/home/test/Desktop/for_test/pcap_resolve/Working.txt'
src_file = '/home/test/Desktop/for_test/pcap_resolve/'
dst_dir = '/home/test/Desktop/for_test/pcap_filter/'

'''
def copyfile(srcfile, dstpath, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + dstfile)  # 复制文件
        #print("copy %s -> %s" % (srcfile, dstpath + fname))

def main(work_dir):
    work_file = datetime.now().strftime("%Y%m%d%H%M%S")
    dirName = './' + work_file
    if os.path.exists(dirName):
        print("Error: Dir exists!")
    else:
        os.mkdir(dirName)

    log_n = -1

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    while True:
        df = pd.read_csv(dst_dir + 'log.csv')
        for row in df.itertuples():
            time.sleep(1)
            if row[0] > log_n:
                #print(row[1])
                log_n = row[0]
                work_now_time = datetime.now().strftime("%Y%m%d%H%M%S")
                copyfile(src_file + str(row[1]) + '.list', dirName + '/', work_now_time + '.list')
                
                #获取resolve的文件后记得记录相关数据
                
                data = data_set(dirName + '/'  + work_now_time + '.list')
                idcs = np.random.permutation(len(data))
                t_data = CustomSubset_two(data, idcs[:])
                test_data = DataLoader(t_data, batch_size=1, shuffle=True)
                for x, y in test_data:
                    with torch.no_grad():
                        x = x.to(device)
                        pred = model(x)
                        predicted = classes[pred[0].argmax(0)]
                        if predicted == 'Normal':
                            print("The program is running normally")
                        else:
                            print(f'The program appears to be under attack! The attack type prediction isPredicted: "{predicted}"')

        copyfile(src_file + work_dir + '.csv', dst_dir, 'log.csv')

if __name__ == "__main__":
    with open(work_log) as f:
        work_dir = f.read()
    f.close()

    work_file = src_file + work_dir + '.csv'
    copyfile(work_file, dst_dir, 'log.csv')
    main(work_dir)
'''


def copyfile(srcfile, dstpath, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + dstfile)  # 复制文件
        #print("copy %s -> %s" % (srcfile, dstpath + fname))

def main(work_dir):
    work_file = datetime.now().strftime("%Y%m%d%H%M%S")
    dirName = '/home/test/Desktop/for_test/pcap_filter/' + work_file
    if os.path.exists(dirName):
        print("Error: Dir exists!")
    else:
        os.mkdir(dirName)

    log_n = -1

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    while True:
        df = pd.read_csv(dst_dir + 'log.csv')
        for row in df.itertuples():
            time.sleep(1)
            if row[0] > log_n:
                #print(row[1])
                log_n = row[0]
                work_now_time = datetime.now().strftime("%Y%m%d%H%M%S")
                copyfile(src_file + str(row[1]) + '.list', dirName + '/', work_now_time + '.list')
                tcpnum = 0
                udpnum = 0
                icmpnum = 0
                #获取resolve的文件后记得记录相关数据
                list = pd.read_csv(dirName + '/' + work_now_time + '.list', sep = ' ')
                for index, rows in list.iterrows():
                    if rows[1] == 'tcp':
                        tcpnum += 1
                    elif rows[1] == 'udp':
                        udpnum += 1
                    else:
                        icmpnum += 1

                result = pd.read_csv(dst_dir + 'result.csv')
                result.loc[0] = int(result.loc[0]) + tcpnum
                result.loc[1] = int(result.loc[1]) + udpnum
                result.loc[2] = int(result.loc[2]) + icmpnum

                data = data_set(dirName + '/' + work_now_time + '.list')
                idcs = np.random.permutation(len(data))
                t_data = CustomSubset_two(data, idcs[:])
                test_data = DataLoader(t_data, batch_size=1, shuffle=True)

                dosnum = 0
                normnum = 0
                probnum = 0
                r2lnum = 0
                u2rnum = 0

                for x, y in test_data:
                    with torch.no_grad():
                        x = x.to(device)
                        pred = model(x)
                        predicted = classes[pred[0].argmax(0)]
                        if predicted == 'Dos':
                            print(f'The program appears to be under attack! The attack type prediction isPredicted: "{predicted}"')
                            dosnum += 1
                        elif predicted == 'Normal':
                            print(
                                f'The program appears to be under attack! The attack type prediction isPredicted: "{predicted}"')
                            normnum += 1
                        elif predicted == 'Probe':
                            print(
                                f'The program appears to be under attack! The attack type prediction isPredicted: "{predicted}"')
                            probnum += 1
                        elif predicted == 'R2L':
                            print(
                                f'The program appears to be under attack! The attack type prediction isPredicted: "{predicted}"')
                            r2lnum += 1
                        else:
                            print(
                                f'The program appears to be under attack! The attack type prediction isPredicted: "{predicted}"')
                            u2rnum += 1
                            #print(f'The program appears to be under attack! The attack type prediction isPredicted: "{predicted}"')

                result.loc[3] = int(result.loc[3]) + dosnum
                result.loc[4] = int(result.loc[4]) + normnum
                result.loc[5] = int(result.loc[5]) + probnum
                result.loc[6] = int(result.loc[6]) + r2lnum
                result.loc[7] = int(result.loc[7]) + u2rnum
                result.to_csv('/home/test/Desktop/for_test/pcap_filter/result.csv', index=False)

        copyfile(src_file + work_dir + '.csv', dst_dir, 'log.csv')

if __name__ == "__main__":
    with open(work_log) as f:
        work_dir = f.read()
    f.close()

    work_file = src_file + work_dir + '.csv'
    copyfile(work_file, dst_dir, 'log.csv')
    main(work_dir)



