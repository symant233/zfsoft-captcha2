import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras


def func(x): return x - 48 if x <= 57 else x - 87 if x <= 110 else x - 88
# func(ord(x)) 0->0 a->10 z->35


def train(data, target, model_save):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(21, 16)),         # 一维化 336
        keras.layers.Dense(21*16, activation=tf.nn.relu),   # 隐藏层
        keras.layers.Dense(36, activation='softmax')        # 输出层 36个可能的值
    ])
    model.compile(optimizer='rmsprop',                      # 优化
                  loss='sparse_categorical_crossentropy',   # 损失函数
                  metrics=['accuracy'])                     # 用准确率衡量
    model.fit(data, target, batch_size=128, epochs=36)
    # 梯度算法37次减少loss, acc接近0.99
    model.save(model_save)


def split_pic(img):
    """img = Image.open('captcha.png')
    returns a dict using np.asarray()
    """
    img = img.convert('L').convert('1')
    x_size, y_size = img.size                      # 72 * 22
    y_size -= 5                                    # 17
    piece = (x_size-24) / 8                        # 6
    centers = [4+piece*(2*i+1) for i in range(4)]

    ar = []
    for i, center in enumerate(centers):
        single_pic = img.crop(
            (center-(piece+2), 1, center+(piece+2), y_size))
        ar.append(np.asarray(single_pic, dtype='int8'))
    return ar


def _load_data(folder):
    count = 0
    imgs = os.listdir(folder)
    length = len(imgs)*4  # 49张图片(full)*4
    label = np.zeros(length, dtype="int8")
    data = np.zeros((length, 21, 16), dtype="int8")
    # * 分配三维空数组, data.shape = (1200, 21, 16)

    for img_name in imgs:
        img = Image.open('%s/%s' % (folder, img_name)
                         ).convert('L').convert('1')
        for i, single in enumerate(split_pic(img)):
            data[count, :] = single
            # ? 将(21*16)pixel的图片转成灰度图像数组, 像MNIST那样
            alpha = img_name.split('.')[0][i]
            label[count] = func(ord(alpha))
            count += 1
    return data, label


if __name__ == "__main__":
    model_file = './model/Model_tf.net'

    print('Training...')
    x_data, y_data = _load_data('./train_data/full/')
    train(x_data, y_data, model_file)

    print('\nTesting...')
    folder = './train_data/test_sets/'
    model = keras.models.load_model(model_file)
    x, y = _load_data(folder)
    model.evaluate(x, y)
