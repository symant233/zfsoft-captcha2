### ZFsoft `[jwgl|jxgl]` system captcha recognition v2

Inspired by [AHU-JiaoWu](https://github.com/AHU-HUI/AHU-JiaoWu),
Modified from `scikit-learn` into `tensorflow`. Add more comments for tutorial use.

[![GitHub repo size](https://img.shields.io/github/repo-size/symant233/zfsoft-captcha2)](https://github.com/symant233/zfsoft-captcha2/archive/master.zip) [![GitHub](https://img.shields.io/github/license/symant233/zfsoft-captcha2)](https://github.com/symant233/zfsoft-captcha2/blob/master/LICENSE) [![GitHub All Releases](https://img.shields.io/github/downloads/symant233/zfsoft-captcha2/total)](https://github.com/symant233/zfsoft-captcha2/releases)

### Usage
```bash
pip install -r requirements.txt -U
python trainer.py # train your own model
python predictor.py  # 预测'./predict/'下的图片
# new
python app.py # flask sever to predict incoming data
```

Target image examples: ![](./predict/0jm8.png) ![](./data/test_sets/r6xe.png) ![](./data/train/5yeg.png)

### Result accuracy
```bash
# train
Epoch 36/36
1200/1200 [==============================] - 0s 47us/sample - loss: 0.0321 - acc: 0.9967
# test
196/196 [==============================] - 0s 476us/sample - loss: 0.1501 - acc: 0.9643
```

### Todo List
 - ~~Add some test sets.~~ (Done)
 - *More comments (for tutorial).*
 - ~~run a flask sever to predict incoming pic data~~ (Done)
 - image_splite improve.
 - ~~`unitest` & `travis-ci`~~ (Abort)
 - command line argv support

### Tutorial
1. retrieve some picture
2. tag the right code of each captcha pics
3. write splite codes 
4. splite code (let output layer 36 [0-9, a-z])
5. using tensorflow train the tagged data<small>[*]</small>
6. save the model for future uses
7. predict given pic with that model

> Step 5 needs some basic tensorflow skills, you can learn these in the link below:
 - [Google Machine Learnig Crash Course](https://developers.google.com/machine-learning/crash-course/prereqs-and-prework) /
 - [Tensorflow official tutorial](https://www.tensorflow.org/tutorials) /
*均支持简体中文*

### Flask sever [optional usage]
> This is optional, you can use `predictor.py` alone by adding pictures into the `predict` folder. Or transmit picture data stream to the flask server.

```bash
python app.py # run this first
```
You can go to http://127.0.0.1:5000/upload see the demo.

Or using some python code below with the `api` (this is POST only) 
```python
import requests

with open('Check.gif', 'rb') as f:
    stream = f.read()

r = requests.post('http://127.0.0.1:5000/api', data=stream)
if r.status_code == 200:
    print(r.text)
```

### Licence
Due to the licence lacunae of original project, I added [MIT LICENCE](https://github.com/symant233/zfsoft-captcha2/blob/master/LICENSE) to my codes. The split picture codes' copyright are belong to @AHU-HUI.

```bash
bash@zfsoft-captcha2$ tree
├── app.py                # flask sever (predictor)
├── model
│   └── Model_tf.net      # tensorflow model created by trainer
├── predict/              # predict folder
├── process   
│   └── split_code_imgs.py
├── requirements.txt      
├── trainer.py            # train and test
├── predictor.py          # predictor
├── data
│   ├── train/            # train images
│   └── test_sets/        # test images
├── README.MD
└── LICENSE
```