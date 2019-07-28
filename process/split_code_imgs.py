import os
from PIL import Image


def img_to_single(train_data_full__folder, train_data_single_folder):
    """
    将完整验证码图片切分成单个字符的图片
    :param train_data_full__folder: 完整图片文件夹，图片事先通过request批量抓取
    :param train_data_single_folder: 切分后图片保存文件夹
    :return: None
    """
    images = os.listdir(train_data_full__folder)
    for img in images:
        image = Image.open('%s/%s' % (train_data_full__folder, img)).convert("L")
        x_size, y_size = image.size  # 72 27
        y_size -= 5 # 22
        # y from 1 to y_size-5
        # x from 4 to x_size-18
        piece = (x_size-24) / 8
        centers = [4+piece*(2*i+1) for i in range(4)]
        pre = img.split('.')[0]
        for i, center in enumerate(centers):
            image.crop(
                (center-(piece+2), 1, center+(piece+2), y_size)
            ).save(
                '%s/%s-%s.png' % (train_data_single_folder, pre, i)
            )


img_to_single('../train_data_full', '../train_data_single')
