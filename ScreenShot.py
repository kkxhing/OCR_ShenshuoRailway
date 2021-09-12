#!/usr/bin/python
# -*- coding=utf-8 -*-
# 截图ScreenShot.py

import ctypes
import win32gui
from PIL import ImageGrab
import win32con
from ctypes import wintypes
import ctypes,os,time


def screenShot():
    # 获取窗口句柄
    hwnd = win32gui.FindWindow("WeChatMainWndForPC", "微信") #此处针对特定程序进行修改
    if not hwnd:
        print('window not found!')
    else:
        print(hwnd)

    # 获取特定程序位置信息
    def get_window_rect(hwnd):
        try:
            f = ctypes.windll.dwmapi.DwmGetWindowAttribute
        except WindowsError:
            f = None
        if f:
            rect = ctypes.wintypes.RECT()
            DWMWA_EXTENDED_FRAME_BOUNDS = 9
            f(ctypes.wintypes.HWND(hwnd),
              ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
              ctypes.byref(rect),
              ctypes.sizeof(rect)
              )
            return rect.left, rect.top, rect.right, rect.bottom

    # win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # 强行显示界面后才好截图
    win32gui.SetForegroundWindow(hwnd)  # 强制将窗口提到最前

    #  裁剪得到全图
    game_rect = get_window_rect(hwnd)
    src_image = ImageGrab.grab(game_rect)
    # src_image = ImageGrab.grab((game_rect[0] + 9, game_rect[1] + 190, game_rect[2] - 9, game_rect[1] + 190 + 450))

    localtime = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 时间戳
    src_image_path = ('PrtSrc' + localtime + '.jpg')
    print(src_image_path)
    src_image.save("./img/"+'PrtSrc' + localtime + '.jpg')
    print("截图时间:" + localtime)
    #src_image.show();


    return src_image_path;

if __name__=='__main__':
    screenShot();
