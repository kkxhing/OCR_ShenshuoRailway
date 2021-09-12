import os
import time
from PIL import Image
import keras.backend as K
import numpy as np
from keras.layers import Flatten, BatchNormalization, Permute, TimeDistributed, Dense, Bidirectional, LSTM
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import SGD, Adam, Adadelta, RMSprop

image_root = './img_cut/'
model_path = './model/crnn_model_weights.hdf5'

img_h = 32
char_file = './char_std_5990.txt'
char = ''
with open(char_file, encoding='utf-8') as f:
    for ch in f.readlines():
        ch = ch.strip('\r\n')
        char = char + ch

char = char[1:] + ' '
n_class = len(char)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def crnn_network(height=img_h, nclass=n_class):

    input = Input(shape=(height, None, 1), name='the_input')
    # CNN
    convolution = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv_1')(input)
    convolution = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool_1')(convolution)
    convolution = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv_2')(convolution)
    convolution = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool_2')(convolution)
    convolution = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv_3')(convolution)
    convolution = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv_4')(convolution)

    convolution = ZeroPadding2D(padding=(0, 1))(convolution)
    convolution = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool_3')(convolution)

    convolution = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv_5')(convolution)
    convolution = BatchNormalization(axis=3)(convolution)
    convolution = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv_6')(convolution)
    convolution = BatchNormalization(axis=3)(convolution)
    convolution = ZeroPadding2D(padding=(0, 1))(convolution)
    convolution = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool_4')(convolution)
    convolution = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv_7')(convolution)

    # m的输出维度为(h, w, c) -> (1, w/4, 512) 转换 (w, b, c) = (seq_len, batch, input_size)
    convolution = Permute((2, 1, 3), name='permute')(convolution)
    convolution = TimeDistributed(Flatten(), name='timedistrib')(convolution)

    # RNN
    recurrent = Bidirectional(LSTM(256, return_sequences=True), name='b_lstm1')(convolution)
    recurrent = Dense(256, name='b_lstm1_out', activation='linear')(recurrent)
    recurrent = Bidirectional(LSTM(256, return_sequences=True), name='b_lstm2')(recurrent)
    y_pred = Dense(nclass, name='b_lstm2_out', activation='softmax')(recurrent)

    basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])

    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer="adadelta", metrics=['accuracy'])
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='rmsprop', metrics=['accuracy'])

    # model.summary()

    return model, basemodel


def predict(img_path, model):

    img = Image.open(img_path)
    img = img.convert('L')

    scale = img.size[1] * 1.0 / 32
    w = int(img.size[0] / scale)
    img = img.resize((w, 32), Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    X = img.reshape((32, w, 1))
    X = np.array([X])
    y_pred = model.predict(X)
    y_pred = y_pred[:, :, :]
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :]
    out_s = u''.join([char[x] for x in out[0]])

    return out_s


if __name__ == '__main__':

    model, basemodel = crnn_network()
    if os.path.exists(model_path):
        basemodel.load_weights(model_path)
        # basemodel.summary()

    files = sorted(os.listdir(image_root))
    for file in files:
        t = time.time()
        image_path = os.path.join(image_root, file)
        print("ocr image is %s" % image_path)
        out = predict(image_path, basemodel)
        print("It takes time : {}s".format(time.time() - t))
        print("result ：%s" % out)

