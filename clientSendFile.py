#!/usr/bin/python
# -*- coding=utf-8 -*-
# 客户端client.py

import win32gui
from PIL import ImageGrab
import win32con
from ctypes import wintypes
import ctypes
import socket
import os,time
import struct
import sys
import ScreenShot

image_root = './img/'



'''客户端发送图片数据到服务器'''
def sock_client_image(src_image_path):

    host = socket.gethostname()  # 获取本地主机名
    port = 12344  # 设置端口号
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 创建 socket 对象
        s.connect((host, port))  # 连接服务器,服务器和客户端都在一个系统下时使用的ip和端口
        #s.connect(('192.168.56.101', 1233))  # 服务器和客户端在不同的系统或不同的主机下时使用的ip和端口，首先要查看服务器所在的系统网卡的ip
        # s.connect(('127.0.0.1', 6666))  #服务器和客户端都在一个系统下时使用的ip和端口
    except socket.error as msg:
        print(msg)
        print(sys.exit(1))
    while True:
        #print(src_image_path)
        filepath=src_image_path
        #filepath = input('input the file: ')  # 输入当前目录下的图片名 xxx.jpg
        fhead = struct.pack(b'128sq', bytes(os.path.basename(filepath), encoding='utf-8'),
                            os.stat(filepath).st_size)  # 将xxx.jpg以128sq的格式打包
        s.send(fhead)

        fp = open(filepath, 'rb')  # 打开要传输的图片
        while True:
            data = fp.read(1024)  # 读入图片数据
            if not data:
                print('{0} send over...'.format(filepath))
                break
            s.send(data)  # 以二进制格式发送图片数据
            #s.send(send_data.encode("gbk"))  # 发送数据
        s.close()
        break    #循环发送

#客户端发送识别数据到服务器
def sock_client_data():
    host = socket.gethostname()  # 获取本地主机名
    port = 12344  # 设置端口号
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
        s.connect((host, port))  # 连接服务器,服务器和客户端都在一个系统下时使用的ip和端口
        # s.connect(('192.168.56.101', 1233))  # 服务器和客户端在不同的系统或不同的主机下时使用的ip和端口，首先要查看服务器所在的系统网卡的ip
        # s.connect(('127.0.0.1', 6666))  #服务器和客户端都在一个系统下时使用的ip和端口
    except socket.error as msg:
        print(msg)
        print(sys.exit(1))


if __name__ == '__main__':

    sock_client_image(r'.\img\PrtSc01.jpg')