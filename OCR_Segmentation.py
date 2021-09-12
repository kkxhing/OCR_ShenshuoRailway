import os
import time
import cv2


def get_img(input_dir):
    img_paths = []
    for (path, dirname, filenames) in os.walk(input_dir):
        for filename in filenames:
            img_paths.append(path+'/'+filename)
    print("img_paths:", img_paths)
    return img_paths


def threshold_demo(image):                          # 全局阈值
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    print("threshold value %s" % ret)
    return binary


def cut_img(img_paths, output_dir):
    scale = len(img_paths)
    for i, img_path in enumerate(img_paths):
        a = "#" * int(i/1000)
        b = "." * (int(scale / 1000)-int(i / 1000))
        c = (i / scale) * 100
        time.sleep(0.2)
        print('正在处理图像： %s' % img_path.split('/')[-1])
        img = cv2.imread(img_path)
        weight = img.shape[1]
        if weight > 1300:
            #cropImg = img[414: 434, 477: 602]       # 裁剪【y1,y2：x1,x2】
            cropImg_1 = img[510: 527, 485: 598]  # SO2标干
            cropImg_2 = img[574:591, 486:600]  # SO2折算
            ''' cropImg_1=img[221: 239, 515: 564] #SO2标干
                cropImg_2=img[252：270，486：600] #SO2折算
                            cropImg_3=img[] #NOx标干
                            cropImg_4=img[] #NOx折算
                            cropImg_5=img[] #颗粒物原始
                            cropImg_6=img[] #颗粒物工况
                            cropImg_7=img[] #颗粒物标干
                            cropImg_8=img[] #颗粒物折算
                            cropImg_9=img[] #O2标干
                            cropImg_10=img[] #烟气湿度
                            cropImg_11=img[] #烟气温度
                            cropImg_12=img[] #烟气静压
                            cropImg_13=img[] #烟气流速
                            cropImg_14=img[] #标干流量
                            cropImg_15=img[] #冷凝器温度
                            cropImg_16=img[] #探头温度
                            cropImg_17=img[] #取样管温度
                        '''
            #for i in range(1,18)
            for i in range(1,3):
                corpImg=locals()["cropImg_"+str(i)]
                cropImg = threshold_demo(corpImg)  # 灰度化
                #cv2.imwrite(output_dir + '/' + img_path.split('/')[-1], cropImg)
                cv2.imwrite(output_dir + '/' + 'PrtSc'+str(i)+'.jpg', cropImg)
            #cropImg= img[221: 239, 515: 564]  # SO2标干

            # cropImg = cv2.resize(cropImg, None, fx=0.5, fy=0.5,
                                 #interpolation=cv2.INTER_CUBIC) #缩小图像
            #cropImg = threshold_demo(cropImg)    # 灰度化
            #cv2.imwrite(output_dir + '/' + img_path.split('/')[-1], cropImg)
        else:
            cropImg_01 = img[414: 434, 477: 602]
            cropImg_01 = threshold_demo(cropImg_01)
            cv2.imwrite(output_dir + '/'+img_path.split('/')[-1], cropImg_01)
        print('{:^3.3f}%[{}>>{}]'.format(c, a, b))

# 476, 221
# 607, 241



if __name__ == '__main__':
    output_dir = "./img_cut"           # 保存截取的图像目录
    input_dir = "./img"                # 读取图片目录表
    img_paths = get_img(input_dir)
    print('图片获取完成 。。。！')
    cut_img(img_paths, output_dir)
