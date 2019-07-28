### ZFsoft `[jwgl|jxgl]` system captcha recognition v2

Inspired by https://github.com/AHU-HUI/AHU-JiaoWu
Modified from scikit-learn to tensorflow.

![GitHub repo size](https://img.shields.io/github/repo-size/symant233/zfsoft-captcha2) ![GitHub](https://img.shields.io/github/license/symant233/zfsoft-captcha2) ![GitHub All Releases](https://img.shields.io/github/downloads/symant233/zfsoft-captcha2/total)

### Usage
```bash
pip install -r requirements.txt -U
# 将需要预测的图片放入'./predict/'下
python tensor.py
```
> 训练需自行取消 `tensor.py` 中的 `__main__` 注释部分

### Todo List
 - ~~Add some test sets.~~
 - *More comments.*
 - image_splite improve.

### Licence
Due to the licence lacunae of original project, I added [MIT LICENCE](https://github.com/symant233/zfsoft-captcha2/blob/master/LICENSE) to my codes. The contents in `./process/[count_char.py/splite_code_imgs.py]` and `./train_data/[full/single/right_code.txt]` 's copyright belongs to @AHU-HUI.

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