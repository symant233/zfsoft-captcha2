from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_data():
    kv_dict = {}
    with open('./train_data/right_code.txt') as f:
        for pre, answers in enumerate(f):
            answers = answers.strip()
            answers = map(lambda x: x - 48 if x <= 57 else x - 87 if x <= 110 else x - 88, map(ord, answers))
            for i, v in enumerate(answers):
                kv_dict['%s-%d.png' % (pre, i)] = v
        "{'0-0.png': 26, ...} 0->0 a->10 r->26"

    folder = './train_data/single'
    imgs = kv_dict.keys()
    length = len(imgs) # 1200张图片(single)
    data = np.zeros((length, 21 , 16), dtype="int8")
    # * 分配三维空数组, data.shape = (1200, 21, 16)
    label = np.zeros(length, dtype="int8")

    for index, img_name in enumerate(imgs):
        img = Image.open('%s/%s' % (folder, img_name)).convert('L').convert('1')
        data[index, :] = np.asarray(img, dtype="int8")
        # ? 将(21*16)pixel的图片转成灰度图像数组, 像MNIST那样
        label[index] = kv_dict[img_name]
    return data, label


def train(data, target, model_save):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(21, 16)),
        keras.layers.Dense(21*16, activation=tf.nn.relu),
        keras.layers.Dense(36, activation='softmax')
    ])
    model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(data, target, batch_size=128, epochs=37)
    model.save(model_save)


def analyse(file_name, model):
    func = lambda x: x + 48 if x <= 9 else x + 87 if x <= 23 else x + 88
    image = Image.open('./predict/%s' % file_name).convert('L').convert('1')

    x_size, y_size = image.size
    y_size -= 5
    piece = (x_size-24) / 8
    centers = [4+piece*(2*i+1) for i in range(4)]

    result = []
    data = np.zeros((1, 21, 16), dtype="int8")
    # for i, box in enumerate(boxs):
    for i, center in enumerate(centers):
        single_pic = image.crop((center-(piece+2), 1, center+(piece+2), y_size))
        # single_pic = image.crop(box)
        data[0]= np.asarray(single_pic, dtype="int8")
        answer = model.predict(data)
        answer = np.argmax(answer)
        result.append(chr(func(answer)))

    return result


if __name__ == '__main__':
    model_file = './model/Model_tf.net'
    
    print('Training...')
    x_data, y_data = load_data()
    train(x_data, y_data, model_file)

    # print('Predict...')
    # model = keras.models.load_model(model_file)
    # import os 
    # for file_name in os.listdir('predict'):
    #     r = analyse(file_name, model)
    #     print(file_name, ':', r)
