from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras


def _load_data():
    "trainer function, return train img data and label"
    kv_dict = {}
    def func(x): return x - 48 if x <= 57 else x - 87 if x <= 110 else x - 88
    with open('./train_data/right_code.txt') as f:
        for pre, answers in enumerate(f):
            answers = answers.strip()  # rh0j like
            answers = map(func, map(ord, answers))
            for i, v in enumerate(answers):
                kv_dict['%s-%d.png' % (pre, i)] = v
    "lambda 将键值对应为 {'0-0.png': 26, ...} 0->0 a->10 z->35"

    folder = './train_data/single'
    imgs = kv_dict.keys()       # * imgs -> ['0-0.png', '0-1.png', ...]
    length = len(imgs)          # * 1200张图片(single)
    label = np.zeros(length, dtype="int8")
    data = np.zeros((length, 21, 16), dtype="int8")
    # * 分配三维空数组, data.shape = (1200, 21, 16)

    for index, img_name in enumerate(imgs):
        img = Image.open('%s/%s' % (folder, img_name)
                         ).convert('L').convert('1')
        data[index, :] = np.asarray(img, dtype="int8")
        # ? 将(21*16)pixel的图片转成灰度图像数组, 像MNIST那样
        label[index] = kv_dict[img_name]
        # ! 注意不能用kv_dict.values() 需要是numpy对象
    return data, label


def train(data, target, model_save):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(21, 16)),         # 一维化 336
        keras.layers.Dense(21*16, activation=tf.nn.relu),   # 隐藏层
        keras.layers.Dense(36, activation='softmax')        # 输出层 36个可能的值
    ])
    model.compile(optimizer='rmsprop',                      # 优化
                  loss='sparse_categorical_crossentropy',   # 损失函数
                  metrics=['accuracy'])                     # 用准确率衡量
    model.fit(data, target, batch_size=128, epochs=37)
    # 梯度算法37次减少loss, acc接近0.99
    model.save(model_save)


def test(x, y, model):
    model.evaluate


def analyse(file_name, model) -> dict:
    """先载入保存的model, 传入model对象来进行分析, 
    model = keras.models.load_model('../model/m.net')
    """
    def func(x): return x + 48 if x <= 9 else x + 87 if x <= 23 else x + 88
    image = Image.open('./predict/%s' % file_name).convert('L').convert('1')
    # 先转化为灰度图像,  再转化为[0|1]值图像

    # 切割图片代码
    x_size, y_size = image.size                     # 72 * 22
    y_size -= 5                                     # 17
    piece = (x_size-24) / 8                         # 6
    centers = [4+piece*(2*i+1) for i in range(4)]

    result = []
    data = np.zeros((1, 21, 16), dtype="int8")
    for i, center in enumerate(centers):
        single_pic = image.crop(
            (center-(piece+2), 1, center+(piece+2), y_size))
        data[0] = np.asarray(single_pic, dtype="int8")
        answer = model.predict(data)       # 此时answer->[36] 每个值都是介于0~1间 总和为1
        answer = np.argmax(answer)         # 找出预测最有信心的
        result.append(chr(func(answer)))   # 将这个值对应转换成character
    return result


if __name__ == '__main__':
    model_file = './model/Model_tf.net'

    # print('Training...')
    # x_data, y_data = _load_data()
    # train(x_data, y_data, model_file)

    print('Predicting...')
    model = keras.models.load_model(model_file)
    import os
    for file_name in os.listdir('predict'):  # 遍历目录文件预测
        r = analyse(file_name, model)
        print(file_name, ':', r)
