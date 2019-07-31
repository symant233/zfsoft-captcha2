from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras


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


def analyse(file_name, model) -> dict:
    """先载入保存的model, 传入model对象来进行分析, 
    model = keras.models.load_model('../model/m.net')
    """
    def func(x): return x + 48 if x <= 9 else x + 87 if x <= 23 else x + 88
    image = Image.open('./predict/%s' % file_name).convert('L').convert('1')
    # 先转化为灰度图像,  再转化为[0|1]值图像
    result = []
    data = np.zeros((1, 21, 16), dtype="int8")
    for single in split_pic(image):
        data[0] = single
        answer = model.predict(data)       # 此时answer->[36] 每个值都是介于0~1间 总和为1
        answer = np.argmax(answer)         # 找出预测最有信心的
        result.append(chr(func(answer)))   # 将这个值对应转换成character
    return result


if __name__ == '__main__':
    model_file = './model/Model_tf.net'

    # Train and test see `trainer.py`
    print('Predicting...')
    model = keras.models.load_model(model_file)
    import os
    for file_name in os.listdir('predict'):  # 遍历目录文件预测
        r = analyse(file_name, model)
        print(file_name, ':', r)
