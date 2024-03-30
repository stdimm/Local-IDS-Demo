from scapy.all import *
import time
import os

count = 200

GetedName = '/home/test/Desktop/for_test/pcap_store/Geted_log.txt'
WorkName = '/home/test/Desktop/for_test/pcap_store/Working_log.txt'
work_now_time = datetime.now().strftime("%Y%m%d%H%M%S")
now_time = []


def callback(packet):
    filename = "/home/test/Desktop/for_test/pcap_store/" + work_now_time + "/{0}.pcap".format(now_time[-1])
    o_open_file = PcapWriter(filename, append=True)
    # packet.show()
    o_open_file.write(packet)


def main():
    dirName = '/home/test/Desktop/for_test/pcap_store/' + work_now_time
    with open(GetedName, 'a') as file1:
        file1.write(work_now_time + '\n')
    file1.close()

    with open(WorkName, 'w') as file2:
        file2.write(work_now_time)
    file2.close()

    if os.path.exists(dirName):
        print("Error: Dir exists!")
    else:
        os.mkdir(dirName)
        TxtName = '/home/test/Desktop/for_test/pcap_store/' + work_now_time + '/log.csv'
        with open(TxtName, 'a') as file:
            file.write('DirName' + '\n')
        file.close()
        while True:
            time.sleep(1)
            now_time.append(datetime.now().strftime("%Y%m%d%H%M%S"))
            print(now_time[-1] + '	strat')
            dpkt_input = sniff(iface="ens33", count=count, prn=callback)
            print(now_time[-1] + '	finish')
            with open(TxtName, 'a') as file:
                file.write(now_time[-1] + '\n')
            file.close()


if __name__ == "__main__":
    main()
