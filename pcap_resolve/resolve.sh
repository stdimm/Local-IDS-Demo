#!/bin/bash
zeek -r /home/test/Desktop/for_test/pcap_resolve/test.pcap /home/test/Desktop/for_test/pcap_resolve/test.zeek > /home/test/Desktop/for_test/pcap_resolve/conn.list
sort -n /home/test/Desktop/for_test/pcap_resolve/conn.list > /home/test/Desktop/for_test/pcap_resolve/conn_sort.list
/home/test/Desktop/for_test/pcap_resolve/trafAld /home/test/Desktop/for_test/pcap_resolve/conn_sort.list $1
#python3 kdd.py trafAld.list
#mv kqout "Kdd/${1##*/}.txt"
#echo "pcap2kdd.sh - ${1##*/} - Done."
