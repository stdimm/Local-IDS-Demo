import csv
import os
import shutil
import signal
import subprocess
import threading
from tkinter import StringVar, IntVar
from pathlib import Path
import ttkbootstrap as ttk
from itertools import cycle
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageSequence
from scapy.all import *
import time
import os
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

ROOTPATH = Path(__file__).parent
PATH = Path(__file__).parent / 'assets'
Filter_Path = Path(__file__).parent / 'pcap_filter'
Resolve_Path = Path(__file__).parent / 'pcap_resolve'

count = 200

GetedName = './pcap_store/Geted_log.txt'
WorkName = './pcap_store/Working_log.txt'
work_now_time = datetime.now().strftime("%Y%m%d%H%M%S")
now_time = []
work_log = './pcap_store/Working_log.txt'
src_file = './pcap_store/'
dst_dir = './pcap_resolve/'
model_path = "./model0.pth"
fwork_log = './pcap_resolve/Working.txt'
fsrc_file = './pcap_resolve/'
fdst_dir = './pcap_filter/'


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def callback(packet):
    filename = "./pcap_store/" + work_now_time + "/{0}.pcap".format(now_time[-1])
    o_open_file = PcapWriter(filename, append=True)
    # packet.show()
    o_open_file.write(packet)


def copyfile(srcfile, dstpath, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + dstfile)  # 复制文件


def getp():
    dirName = './pcap_store/' + work_now_time
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
        TxtName = './pcap_store/' + work_now_time + '/log.csv'
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


def resolve():
    with open(work_log) as f:
        work_dir = f.read()
    f.close()

    work_file = src_file + work_dir + '/log.csv'
    copyfile(work_file, dst_dir, 'log.csv')
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
                # print(row[1])
                log_n = row[0]
                work_now_time = datetime.now().strftime("%Y%m%d%H%M%S")
                copyfile(src_file + work_dir + '/' + str(row[1]) + '.pcap', dst_dir, 'test.pcap')
                n = os.system('sh /home/test/Desktop/for_test/pcap_resolve/resolve.sh ' + work_now_time)
                with open(dst_dir + work_file + '.csv', 'a') as file:
                    file.write(work_now_time + '\n')
                file.close()
                # copyfile('./' + work_now_time + '.list', '../pcap_filter/', work_now_time + '.txt')
        # print("End")
        copyfile(src_file + work_dir + '/' + 'log.csv', dst_dir, 'log.csv')


def filter():
    with open(fwork_log) as f:
        fwork_dir = f.read()
    f.close()

    fwork_file = fsrc_file + fwork_dir + '.csv'
    copyfile(fwork_file, fdst_dir, 'log.csv')

    work_file = datetime.now().strftime("%Y%m%d%H%M%S")
    dirName = './pcap_filter/' + work_file
    if os.path.exists(dirName):
        print("Error: Dir exists!")
    else:
        os.mkdir(dirName)

    log_n = -1

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    while True:
        df = pd.read_csv(fdst_dir + 'log.csv')
        for row in df.itertuples():
            time.sleep(1)
            if row[0] > log_n:
                # print(row[1])
                log_n = row[0]
                work_now_time = datetime.now().strftime("%Y%m%d%H%M%S")
                copyfile('./' + str(row[1]) + '.list', dirName + '/', work_now_time + '.list')
                tcpnum = 0
                udpnum = 0
                icmpnum = 0
                # 获取resolve的文件后记得记录相关数据
                list = pd.read_csv(dirName + '/' + work_now_time + '.list', sep=' ')
                for index, rows in list.iterrows():
                    if rows[1] == 'tcp':
                        tcpnum += 1
                    elif rows[1] == 'udp':
                        udpnum += 1
                    else:
                        icmpnum += 1

                result = pd.read_csv('./result.csv')
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
                            print(
                                f'The program appears to be under attack! The attack type prediction isPredicted: "{predicted}"')
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
                            # print(f'The program appears to be under attack! The attack type prediction isPredicted: "{predicted}"')

                result.loc[3] = int(result.loc[3]) + dosnum
                result.loc[4] = int(result.loc[4]) + normnum
                result.loc[5] = int(result.loc[5]) + probnum
                result.loc[6] = int(result.loc[6]) + r2lnum
                result.loc[7] = int(result.loc[7]) + u2rnum
                result.to_csv('./result.csv', index=False)

        copyfile(fsrc_file + fwork_dir + '.csv', fdst_dir, 'log.csv')


class ThreadWorker(threading.Thread):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            subprocess.run(["python", self.target])

    def stop(self):
        self.stop_event.set()


class main(ttk.Frame):

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.root = master
        self.createPage()

    def createPage(self):
        # application images
        self.TCPNUM = IntVar()
        self.TCPNUMstr = StringVar()
        self.UDPNUM = IntVar()
        self.UDPNUMstr = StringVar()
        self.ICMPNUM = IntVar()
        self.ICMPNUMstr = StringVar()
        self.JCQNUM = IntVar()
        self.JCQUSEDNUM = IntVar()
        self.WARNINGNUM = IntVar()
        self.WARNINGCATEGORY = IntVar()
        self.INVNUM = IntVar()
        self.GuanListr = StringVar()
        self.process1 = Thread(target=getp, args=())
        self.process2 = Thread(target=resolve, args=())
        self.process3 = Thread(target=filter, args=())

        self.DosNum = 0
        self.NormNum = 0
        self.ProbNum = 0
        self.R2lNum = 0
        self.U2rNum = 0
        self.threads = []

        self.images = [
            ttk.PhotoImage(
                name='logo',
                file=PATH / 'shieldicons.png'),
            ttk.PhotoImage(
                name='main',
                file=PATH / 'main.png'),
            ttk.PhotoImage(
                name='settings',
                file=PATH / 'settings.png'),
            ttk.PhotoImage(
                name='about',
                file=PATH / 'about.png'),
            ttk.PhotoImage(
                name='comp',
                file=PATH / 'comp.png'),
            ttk.PhotoImage(
                name='test',
                file=PATH / 'tests.png'),
            ttk.PhotoImage(
                name='go',
                file=PATH / 'go.png'),
            ttk.PhotoImage(
                name='stop',
                file=PATH / 'stop.png'),
            ttk.PhotoImage(
                name='back',
                file=PATH / 'back.png'),
            ttk.PhotoImage(
                name='icons',
                file=PATH / 'icons.png'),

        ]
        gif_path = Path(__file__).parent / "assets/running.gif"

        # header
        hdr_frame = ttk.Frame(self, padding=20, bootstyle=SECONDARY)
        hdr_frame.grid(row=0, column=0, columnspan=3, sticky=EW)

        hdr_label = ttk.Label(
            master=hdr_frame,
            image='logo',
            bootstyle=(INVERSE, SECONDARY)
        )
        hdr_label.pack(side=LEFT)

        logo_text = ttk.Label(
            master=hdr_frame,
            text='基于CFL的网络入侵检测系统',
            font=('TkDefaultFixed', 30),
            bootstyle=(INVERSE, SECONDARY)
        )
        # 设置header的长度
        logo_text.pack(side=LEFT, padx=1)

        # action buttons
        action_frame = ttk.Frame(self)
        action_frame.grid(row=1, column=0, sticky=NSEW)

        cleaner_btn = ttk.Button(
            master=action_frame,
            image='main',
            text='主页',
            compound=TOP,
            bootstyle=INFO,
            command=self.mainpage_disp
        )
        cleaner_btn.pack(side=TOP, fill=BOTH, ipadx=1, ipady=50)

        tools_btn = ttk.Button(
            master=action_frame,
            image='settings',
            text='选项',
            compound=TOP,
            bootstyle=INFO,
            command=self.setting_disp
        )
        tools_btn.pack(side=TOP, fill=BOTH, ipadx=0, ipady=50)

        options_btn = ttk.Button(
            master=action_frame,
            image='about',
            text='软件相关信息',
            compound=TOP,
            bootstyle=INFO,
            command=self.about_disp
        )
        options_btn.pack(side=TOP, fill=BOTH, ipadx=0, ipady=30)

        self.pack(fill=BOTH, expand=YES)

        # Main
        MainWidth = 10
        Main = ttk.Frame(self, padding=10)

        # start_page
        start_page = ttk.Labelframe(Main, text='开始界面', padding=10)
        start_page.pack(side=TOP, fill=BOTH, expand=YES)

        # header
        start_page_header = ttk.Frame(start_page, padding=5)
        start_page_header.pack(fill=X)

        lbl = ttk.Label(start_page_header, text='启动后按照保存配置运行,如需更改请在设置界面修改')
        lbl.pack(side=LEFT, fill=X, padx=15)

        btn = ttk.Button(
            master=start_page_header,
            image='icons',
            bootstyle=LINK
        )
        btn.pack(side=RIGHT)

        backpage = ttk.Label(start_page, image='back')

        run_btn = ttk.Button(
            master=start_page,
            image='go',
            text='开始运行',
            compound=TOP,
            bootstyle=LINK,
            command=self.run_disp,
            width=MainWidth,

        )

        stop_btn = ttk.Button(
            master=start_page,
            image='stop',
            text='停止运行',
            compound=TOP,
            bootstyle=LINK,
            command=self.stop_disp,
            width=MainWidth
        )

        with Image.open(gif_path) as im:
            # create a sequence
            sequence = ImageSequence.Iterator(im)
            images = [ImageTk.PhotoImage(s) for s in sequence]
            self.image_cycle = cycle(images)
            # length of each frame
            self.framerate = im.info["duration"]

        self.gif = ttk.Label(start_page, image=next(self.image_cycle), width=MainWidth, compound=TOP)

        self.new_col = ttk.Frame(self, padding=10)

        finger_gest = ttk.Labelframe(
            master=self.new_col,
            text='平台管理信息',
            padding=(15, 10)
        )
        finger_gest.pack(
            side=TOP,
            fill=BOTH,
            expand=YES,
            pady=(10, 0)
        )

        self.GuanListr.set("平台安全状态良好")
        Warninfo = ttk.Label(finger_gest, text='平台安全状态:              ', bootstyle="dark")
        warninfos = ttk.Label(finger_gest, textvariable=self.GuanListr, bootstyle="warning")

        Warninfo.pack(side=LEFT)
        warninfos.pack(side=LEFT)

        # licence info
        lic_info = ttk.Labelframe(self.new_col, text='网络连接分类', padding=50)
        lic_info.pack(side=LEFT, fill=BOTH, expand=YES, pady=(1, 0))
        # lic_info.grid(row = 1, column = 0, sticky = NSEW)
        # .rowconfigure(0, weight=1)
        # lic_info.columnconfigure(0, weight=2)

        self.JCQNUM.set(1)
        self.JCQUSEDNUM.set(1)
        jcq_info = ttk.Frame(lic_info, padding=5)
        jcqlb = ttk.Label(jcq_info, text='监测器/总共:        ', bootstyle="dark")
        jcqlb_ = ttk.Label(jcq_info, text='/', bootstyle="dark")
        jcqnum = ttk.Label(jcq_info, textvariable=self.JCQNUM, bootstyle="warning")
        jcqusednum = ttk.Label(jcq_info, textvariable=self.JCQUSEDNUM, bootstyle="info")
        jcq_info.pack(fill=X)
        jcqlb.pack(side=LEFT)
        jcqusednum.pack(side=LEFT)
        jcqlb_.pack(side=LEFT)
        jcqnum.pack(side=LEFT)

        self.WARNINGNUM.set(0)
        warning_info = ttk.Frame(lic_info, padding=5)
        warnlb = ttk.Label(warning_info, text='警告总数:              ', bootstyle="dark")
        warnings = ttk.Label(warning_info, textvariable=self.WARNINGNUM, bootstyle="warning")
        Categoryp = ttk.Label(warning_info, text='警告类别:              ', bootstyle="dark")
        Categorys = ttk.Label(warning_info, textvariable=self.WARNINGCATEGORY, bootstyle="warning")

        warning_info.pack(fill=X)
        warnlb.pack(side=LEFT)
        warnings.pack(side=LEFT)
        Categorys.pack(side=RIGHT)
        Categoryp.pack(side=RIGHT)

        self.INVNUM.set(0)
        self.Invar = StringVar()
        Inv_Cat = ttk.Frame(lic_info, padding=5)
        cbo = ttk.Combobox(
            master=Inv_Cat,
            textvariable=self.Invar,
            values=['正常流量数量', 'Dos攻击数量', 'Probe攻击数量', 'U2R攻击数量', 'R2L攻击数量'],
            bootstyle="dark"
        )
        ivonum = ttk.Label(Inv_Cat, textvariable=self.INVNUM, bootstyle="warning")

        Inv_Cat.pack(fill=X)
        cbo.current(0)
        cbo.bind('<<ComboboxSelected>>', self.Cateshow)
        cbo.pack(side=LEFT)
        ivonum.pack(side=RIGHT)

        num_protocol = ttk.Labelframe(lic_info, text='网络连接协议数量统计', padding=30)
        num_protocol.pack()
        # TCP统计
        self.tcp = ttk.Progressbar(num_protocol, variable=self.TCPNUM, bootstyle="striped", length=150)
        # self.tcpl = ttk.Label(self.tcp, , bootstyle=(PRIMARY, INVERSE))
        self.tcplb = ttk.Label(
            master=num_protocol,
            textvariable=self.TCPNUMstr,
            # text='TCP 链接数据统计',
            font='Helvetica 8',
            anchor=CENTER,
        )
        # self.tcpl.pack()

        # UDP统计
        self.udp = ttk.Progressbar(num_protocol, variable=self.UDPNUM, bootstyle="striped", length=150)
        # self.udpl = ttk.Label(self.udp, textvariable=self.UDPNUMstr, bootstyle=PRIMARY)
        self.udplb = ttk.Label(
            master=num_protocol,
            textvariable=self.UDPNUMstr,
            # text='UDP 链接数据统计',
            font='Helvetica 8',
            anchor=CENTER,
        )
        # self.udpl.pack()

        # icmp统计
        self.icmp = ttk.Progressbar(num_protocol, variable=self.ICMPNUM, bootstyle="striped", length=150)
        # self.icmpl = ttk.Label(self.icmp, , bootstyle=(PRIMARY, INVERSE))
        self.icmplb = ttk.Label(
            master=num_protocol,
            textvariable=self.ICMPNUMstr,
            font='Helvetica 8',
            anchor=CENTER,
        )
        # self.icmpl.pack()

        self.tcplb.pack(side=BOTTOM)
        self.tcp.pack(fill=BOTH, pady=5, padx=50, side=BOTTOM)
        self.udplb.pack(side=BOTTOM)
        self.udp.pack(fill=BOTH, pady=5, padx=50, side=BOTTOM)
        self.icmplb.pack(side=BOTTOM)
        self.icmp.pack(fill=BOTH, pady=5, padx=50, side=BOTTOM)
        self.stop_btn = stop_btn
        self.run_btn = run_btn
        self.backpage = backpage
        backpage.pack(fill=X)
        run_btn.pack(fill=X)

        # Setting
        Setting = ttk.Notebook(self)
        settingpage = ttk.Frame(self, padding=100)
        wt_scrollbar = ttk.Scrollbar(settingpage)
        wt_scrollbar.pack(side=RIGHT, fill=Y)
        wt_scrollbar.set(0, 1)

        wt_canvas = ttk.Canvas(
            master=settingpage,
            relief=FLAT,
            borderwidth=0,
            selectborderwidth=0,
            highlightthickness=0,
            yscrollcommand=wt_scrollbar.set
        )
        wt_canvas.pack(side=LEFT, fill=BOTH)

        # adjust the scrollregion when the size of the canvas changes
        wt_canvas.bind(
            sequence='<Configure>',
            func=lambda e: wt_canvas.configure(
                scrollregion=wt_canvas.bbox(ALL))
        )
        wt_scrollbar.configure(command=wt_canvas.yview)
        scroll_frame = ttk.Frame(wt_canvas)
        wt_canvas.create_window((0, 0), window=scroll_frame, anchor=NW)

        radio_options = [
            '数据流量实时抓取', '模型分析脚本开启', '流量数据保存',
            '流量处理数据保存', '流量链条保存',
        ]
        interface_options = [
            '界面高亮', '动态模糊', '动态动画',
            '处理信息更新', '设置自动保存',
        ]

        network = ttk.Labelframe(
            master=scroll_frame,
            text='网络设置',
            padding=(20, 10)
        )
        network.pack(fill=BOTH, expand=YES, padx=100, pady=10)

        interface = ttk.Labelframe(
            master=scroll_frame,
            text='界面设置',
            padding=(20, 10)
        )
        interface.pack(fill=BOTH, padx=100, pady=10, expand=YES)

        # add radio buttons to each label frame section
        for section in [network]:
            for opt in radio_options:
                cb = ttk.Checkbutton(section, text=opt, state=NORMAL)
                cb.invoke()
                cb.pack(side=TOP, pady=2, fill=X)

        for section in [interface]:
            for opt in interface_options:
                cb = ttk.Checkbutton(section, text=opt, state=NORMAL)
                cb.invoke()
                cb.pack(side=TOP, pady=2, fill=X)

        Setting.add(settingpage, text='设置')

        # About
        About = ttk.Frame(self)

        # result cards
        cards_frame = ttk.Frame(
            master=About,
            name='cards-frame',
            bootstyle=SECONDARY
        )
        cards_frame.pack(fill=BOTH, expand=YES)

        # privacy card
        priv_card = ttk.Frame(
            master=cards_frame,
            padding=1,
        )
        priv_card.pack(side=LEFT, fill=BOTH, padx=(10, 5), pady=10)

        priv_container = ttk.Frame(
            master=priv_card,
            padding=100,
        )
        priv_container.pack(fill=BOTH, expand=YES)

        priv_lbl = ttk.Label(
            master=priv_container,
            image='comp',
            text='关于本平台代码所有者',
            compound=TOP,
            anchor=CENTER
        )
        priv_lbl.pack(fill=BOTH, padx=200, pady=(40, 0))

        ttk.Label(
            master=priv_container,
            textvariable='priv_lbl',
            bootstyle=PRIMARY
        ).pack(pady=(0, 20))
        self.setvar('priv_lbl', '2023.4.1 本代码用于《应用公平聚类联邦学习的入侵检测系统》设计展示用')

        # user notification
        note_frame = ttk.Frame(
            master=About,
            bootstyle=SECONDARY,
            padding=40
        )
        note_frame.pack(fill=BOTH)

        note_msg = ttk.Label(
            master=note_frame,
            text='本代码后续正在开发中   ing',
            anchor=CENTER,
            font=('Helvetica', 12, 'italic'),
            bootstyle=(INVERSE, SECONDARY)
        )
        note_msg.pack(fill=BOTH)

        self.Setting = Setting
        self.About = About
        self.Main = Main
        self.currentPage = Setting
        self.currentPage.grid(row=1, column=1, sticky=NSEW, pady=(25, 0))

    def setting_disp(self):
        self.currentPage.grid_forget()
        self.new_col.grid_forget()
        self.currentPage = self.Setting
        self.currentPage.grid(row=1, column=1, sticky=NSEW, pady=(25, 0))

    def mainpage_disp(self):
        self.currentPage.grid_forget()
        self.currentPage = self.Main
        self.currentPage.grid(row=1, column=1, sticky=NSEW, pady=(25, 0))
        self.new_col.grid(row=1, column=2, sticky=NSEW)
        self.TCPNUM.set(0)
        self.UDPNUM.set(0)
        self.ICMPNUM.set(0)
        self.TCPNUMstr.set("TCP 链接数据统计   0%")
        self.UDPNUMstr.set("UDP 链接数据统计   0%")
        self.ICMPNUMstr.set("ICMP 链接数据统计   0%")

    def about_disp(self):
        self.currentPage.grid_forget()
        self.new_col.grid_forget()
        self.currentPage = self.About
        self.currentPage.grid(row=1, column=1, sticky=NSEW, pady=(25, 0))

    def run_disp(self):
        self.run_btn.pack_forget()
        self.backpage.pack_forget()
        self.tcptimer = self.after(500, self.next_numupdate)
        self.gif.pack(fill="x")
        self.rungif = self.after(self.framerate, self.next_frame)
        self.stop_btn.pack(fill="x")
        self.start_threads()

    def stop_disp(self):
        self.gif.pack_forget()
        self.after_cancel(self.tcptimer)
        self.after_cancel(self.rungif)
        self.stop_btn.pack_forget()
        self.backpage.pack(fill="x")
        self.run_btn.pack(fill="x")
        self.stop_threads()

    def next_frame(self):
        """Update the image for each frame"""
        self.gif.configure(image=next(self.image_cycle))
        self.rungif = self.after(self.framerate, self.next_frame)

    def next_numupdate(self):
        result = pd.read_csv('./result.csv')
        tcpnum = int(result.loc[0])
        udpnum = int(result.loc[1])
        icmpnum = int(result.loc[2])
        num = tcpnum + udpnum + icmpnum
        tcpnums = int(tcpnum * 100 / num)
        udpnums = int(udpnum * 100 / num)
        icmpnums = int(icmpnum * 100 / num)
        dos = int(result.loc[3])
        norm = int(result.loc[4])
        prob = int(result.loc[5])
        r2l = int(result.loc[6])
        u2r = int(result.loc[7])
        if (dos - self.DosNum) > 30:
            self.GuanListr.set("平台正在遭受Dos攻击，关注安全警告信息并采取措施！！！！")

        self.DosNum = dos
        self.NormNum = norm
        self.ProbNum = prob
        self.R2lNum = r2l
        self.U2rNum = u2r

        if (self.DosNum / num) > 0.4:
            if self.DosNum > self.NormNum:
                self.GuanListr.set("平台疑似受到Dos攻击，请及时检查相关安全警告信息")
            elif self.DosNum < self.NormNum:
                self.GuanListr.set("平台检测到大量Dos流量，请及时排查设置相关问题")

        self.WARNINGNUM.set(self.DosNum + self.ProbNum + self.R2lNum + self.U2rNum)

        self.TCPNUM.set(tcpnums)
        self.UDPNUM.set(udpnums)
        self.ICMPNUM.set(icmpnums)

        self.TCPNUMstr.set('TCP 链接数据统计  ' + str(self.TCPNUM.get()) + "%")
        self.UDPNUMstr.set('UDP 链接数据统计  ' + str(self.UDPNUM.get()) + "%")
        self.ICMPNUMstr.set('ICMP 链接数据统计  ' + str(self.ICMPNUM.get()) + "%")
        self.tcptimer = self.after(500, self.next_numupdate)

    def Cateshow(self, envent):
        if self.Invar.get() == '正常流量数量':
            self.INVNUM.set(self.NormNum)
            if self.process1.is_alive():
                print("2-1")
                if not self.process2.is_alive():
                    print("2-2")
                    self.process2.start()

        elif self.Invar.get() == 'Dos攻击数量':
            self.INVNUM.set(self.DosNum)
            if self.process1.is_alive():
                print("3-1")
                if not self.process3.is_alive():
                    print("3-2")
                    self.process3.start()

        elif self.Invar.get() == 'Probe攻击数量':
            self.INVNUM.set(self.ProbNum)

        elif self.Invar.get() == 'U2R攻击数量':
            self.INVNUM.set(self.U2rNum)

        elif self.Invar.get() == 'R2L攻击数量':
            self.INVNUM.set(self.R2lNum)

    def start_threads(self):
        # pass
        # 启动三个线程，每个线程运行一个 Python 脚本


        self.process1.start()

        # self.process1.start()
        # time.sleep(3)
        # self.process2.start()
        # time.sleep(3)
        # self.process3.start()

    def stop_threads(self):
        if self.process1.is_alive():
            stop_thread(self.process1)
        if self.process2.is_alive():
            stop_thread(self.process2)
        if self.process3.is_alive():
            stop_thread(self.process3)


if __name__ == '__main__':
    app = ttk.Window("基于CFL的网络入侵检测系统 v1.0", "pulse")
    main(app)
    app.mainloop()
    r = os.system('sh /home/test/Desktop/for_test/result.sh')
    result = pd.read_csv('./result.csv')
    result.loc[0] = 100
    result.loc[1] = 0
    result.loc[2] = 0
    result.loc[3] = 0
    result.loc[4] = 100
    result.loc[5] = 0
    result.loc[6] = 0
    result.loc[7] = 0
    result.to_csv('./result.csv', index=False)

