import time

from PySide2.QtWidgets import QApplication,QMainWindow,QPushButton,QPlainTextEdit,QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from os import times

import ScreenShot
from ScreenShot import *
#import ScreenShot
import OCR_Predict,OCR_Segmentation
import clientSendFile


image_root = './img_cut/'
model_path = './model/crnn_model_weights.hdf5'

def start():
    # 截图
    path = ScreenShot.screenShot() # 返回截图后的路径
    localtime = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 时间戳
    ui.textBrowser.append("截图时间:" + localtime) #提示信息：截图完成

    # 分割成17个小块
    output_dir = "./img_cut"  # 保存截取的图像目录
    input_dir = "./img"  # 读取图片目录表
    img_paths = OCR_Segmentation.get_img(input_dir)
    print('图片获取完成 。。。！')
    OCR_Segmentation.cut_img(img_paths, output_dir)

    # 识别
    model, basemodel = OCR_Predict.crnn_network()
    if os.path.exists(model_path):
        basemodel.load_weights(model_path)
        # basemodel.summary()

    files = sorted(os.listdir(image_root))
    for file in files:
        t = time.time()
        image_path = os.path.join(image_root, file)
        print("ocr image is %s" % image_path)
        out = OCR_Predict.predict(image_path, basemodel)  # 输出的结果

        print("It takes time : {}s".format(time.time() - t))
        print("result ：%s" % out)
        #后续还需将结果封装成字典发送到服务器端

    # 发送数据到服务器数据库
    clientSendFile.sock_client_image("./img/"+path)  # 发送图片数据
    ui.textBrowser.append('图数据发送完成 。。。！')  # 提示信息：数据完成
    #clientSendFile.sock_client_data() #发送识别数据
    #ui.textBrowser.append('识别数据发送完成 。。。！')  # 提示信息：数据完成
def click(flag):
    flag=False;

def loop():
    flag=True;
    while (flag==True):
        start()
        time.sleep(30)
        ui.pushButton3.clicked.connect(click(flag))
        if(flag==False):
            break


app = QApplication([]) #建立application对象
#从文件中加载UI定义
qfile_test=QFile("untitled.ui")
qfile_test.open(QFile.ReadOnly)
qfile_test.close()

#从UI定义中动态创建一个相应的窗口对象
ui=QUiLoader().load(qfile_test) #返回窗口对象，所有组件属于self.ui的属性

localtime = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 时间戳
ui.pushButton.clicked.connect(start)#点击按钮后开始截图.切记！！！方法不能加括号


#runtest=RunTest()
ui.show()#显示窗体
app.exec_() #运行程序

