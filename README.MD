### ZFsoft `[jwgl|jxgl]` system captcha recognition v2

Inspired by [AHU-JiaoWu](https://github.com/AHU-HUI/AHU-JiaoWu),
Modified from `scikit-learn` into `tensorflow`. Add more comments for tutorial use.

[![GitHub repo size](https://img.shields.io/github/repo-size/symant233/zfsoft-captcha2)](https://github.com/symant233/zfsoft-captcha2/archive/master.zip) [![GitHub](https://img.shields.io/github/license/symant233/zfsoft-captcha2)](https://github.com/symant233/zfsoft-captcha2/blob/master/LICENSE) [![GitHub All Releases](https://img.shields.io/github/downloads/symant233/zfsoft-captcha2/total)](https://github.com/symant233/zfsoft-captcha2/releases)

### Usage
```bash
pip install -r requirements.txt -U
# 将需要预测的图片放入'./predict/'下
python tensor.py
```
> 训练需自行取消 `tensor.py` 中的 `__main__` 注释部分

### Todo List
 - ~~Add some test sets.~~
 - *More comments (for tutorial).*
 - run a flask sever to predict incoming pic data
 - image_splite improve.
 - `unitest` & ~~`travis-ci`~~
 - command line argv support

### Tutorial
1. retrieve some picture
2. tag the right code of each captcha pics
3. write splite codes 
4. splite code (let output layer 36 [0~9, a~z])
5. using tensorflow train the tagged data<small>[*]</small>
6. save the model for future uses
7. predict given pic with that model

> Step 5 needs some basic tensorflow skills, you can learn these in the link below:
> - [Google Machine Learnig Crash Course](https://developers.google.com/machine-learning/crash-course/prereqs-and-prework) /
> - [Tensorflow official tutorial](https://www.tensorflow.org/tutorials) /
> *均支持简体中文*

### Licence
Due to the licence lacunae of original project, I added [MIT LICENCE](https://github.com/symant233/zfsoft-captcha2/blob/master/LICENSE) to my codes. The contents' copyright that in `./process/[count_char.py/splite_code_imgs.py]` and `./train_data/[full/single/right_code.txt]` are belong to @AHU-HUI.

```bash
bash@zfsoft-captcha2$ tree
├── model
│   └── Model_tf.net      # tensorflow model created by trainer
├── predict
│   ├── 0jm8.png
│   ├── 1atu.png          # predict folder
│   └── 1r0q.png
├── process
│   ├── count_char.py     
│   └── split_code_imgs.py
├── requirements.txt      
├── tensor.py             # project entry (trainer & predictor)
├── train_data
│   ├── full/
│   ├── single/
│   └── right_code.txt
├── README.MD
└── LICENSE
```