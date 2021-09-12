import ScreenShot
import OCR_Predict,OCR_Segmentation
import clientSendFile
import os,time

image_root = './img_cut/'
model_path = './model/crnn_model_weights.hdf5'

#截图
path=ScreenShot() #返回截图后的路径

#分割成17个小块
output_dir = "./img_cut"  # 保存截取的图像目录
input_dir = "./img"  # 读取图片目录表
img_paths = OCR_Segmentation.get_img(input_dir)
print('图片获取完成 。。。！')
OCR_Segmentation.cut_img(img_paths, output_dir)

#识别
model, basemodel = OCR_Predict.crnn_network()
if os.path.exists(model_path):
    basemodel.load_weights(model_path)
    # basemodel.summary()

files = sorted(os.listdir(image_root))
for file in files:
    t = time.time()
    image_path = os.path.join(image_root, file)
    print("ocr image is %s" % image_path)
    out =OCR_Predict.predict(image_path, basemodel) #输出的结果

    print("It takes time : {}s".format(time.time() - t))
    print("result ：%s" % out)

    #发送数据到服务器数据库
    clientSendFile.sock_client_image(path) #发送图片数据
    #clientSendFile.sock_client_data() #发送识别结果