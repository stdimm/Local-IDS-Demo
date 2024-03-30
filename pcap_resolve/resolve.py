import os
import shutil
from datetime import datetime
import time
import pandas as pd

work_log = '/home/test/Desktop/for_test/pcap_store/Working_log.txt'
src_file = '/home/test/Desktop/for_test/pcap_store/'
dst_dir = '/home/test/Desktop/for_test/pcap_resolve/'

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

    with open(dst_dir + 'resolved.txt', 'a') as file:
        file.write(work_file + '\n')
    file.close()

    with open(dst_dir + 'Working.txt', 'w') as file:
        file.write(work_file)
    file.close()

    with open(dst_dir + work_file + '.csv', 'a') as file:
        file.write('Resolved File' + '\n')
    file.close()

    log_n = -1
    
    while True:
        df = pd.read_csv(dst_dir + 'log.csv')
        for row in df.itertuples():
            time.sleep(1)
            if row[0] > log_n:
                #print(row[1])
                log_n = row[0]
                work_now_time = datetime.now().strftime("%Y%m%d%H%M%S")
                copyfile(src_file + work_dir + '/' + str(row[1]) + '.pcap', dst_dir, 'test.pcap')
                n = os.system('sh /home/test/Desktop/for_test/pcap_resolve/resolve.sh ' + work_now_time)
                with open(dst_dir + work_file + '.csv', 'a') as file:
                    file.write(work_now_time + '\n')
                file.close()
                #copyfile('./' + work_now_time + '.list', '../pcap_filter/', work_now_time + '.txt')
        #print("End")
        copyfile(src_file + work_dir + '/' + 'log.csv', dst_dir, 'log.csv')

if __name__ == "__main__":
    with open(work_log) as f:
        work_dir = f.read()
    f.close()

    work_file = src_file + work_dir + '/log.csv'
    copyfile(work_file, dst_dir, 'log.csv')
    main(work_dir)
