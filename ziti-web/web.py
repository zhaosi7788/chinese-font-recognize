# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from nnets.vgg import vgg
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import colorsys
import random

import string

def get_dominant_color(image):
    # 颜色模式转换，以便输出rgb颜色值
    image = image.convert('RGBA')

    # 生成缩略图，减少计算量，减小cpu压力
    image.thumbnail((200, 200))

    max_score = 0
    dominant_color = 0

    style = "font-family:Arial, Helvetica, sans-serif;"

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue

        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]

        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

        y = (y - 16.0) / (235 - 16)

        # 忽略高亮色
        if y > 0.9:
            continue

        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

    return dominant_color


app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 不使用GPU
inputs = tf.placeholder(tf.float32, shape=[None, None, 3])
example = tf.cast(tf.image.resize_images(inputs, [128, 128]), tf.uint8)
example = tf.image.per_image_standardization(example)
example = tf.expand_dims(example, 0)
output = vgg(example, 9, 1.0)
sess = tf.Session()
tf.train.Saver().restore(sess, 'models/vgg.ckpt')
print("模型重载成功")

'''导入模型'''
@app.route('/')
def about():
    return redirect(url_for('upload'))
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':

        print("已经接收到汉字字体图片。")
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        ramdonnum = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        upload_path = os.path.join(basepath,'static/uploads',ramdonnum+f.filename)  #注意：没有的文件夹一定要先创建，不然会提示没有该路径

        f.save(upload_path)
        print("图片存储完成，存储地址为：")

        real_path = ramdonnum+f.filename
        print(upload_path)
        data= Image.open(upload_path)
        # data=data.resize((128,128))
        get_dominant_color(data)

        # f1 = open(upload_path)
        # real_path = os.path.realpath(f1.name)

        '''跟文件名没有关系'''

        pred = sess.run(output, feed_dict={inputs: data})

        pred = np.squeeze(pred)
        pred = pred.tolist()
        pred = [round(a, 4) for a in pred]
        prob = u"概率为"+str(max(pred)*100)+"%"
        fenlei = ''
        index = pred.index(max(pred))
        '''获取最大值进行判断'''
        if index == 0:
            fenlei = u'隶体'
        elif index == 1:
            fenlei = u'篆体'
        elif index == 2:
            fenlei = u'草体'
        elif index == 3:
            fenlei = u'楷体'
        elif index == 4:
            fenlei = u'楷体（柳体）'
        elif index == 5:
            fenlei = u'楷体（魏碑）'
        elif index == 6:
            fenlei = u'楷体（赵体）'
        elif index == 7:
            fenlei = u'楷体（颜体）'
        elif index == 8:
            fenlei = u'楷体（欧体）'


        print("汉字字体识别完成，识别字体为："+fenlei+"。")
        print("识别准确度为："+str(max(pred)*100)+"%。")

        return render_template('upload.html', result=fenlei,imgpath = real_path,prob=prob)
    else:
     return render_template('upload.html',imgpath ="demo/demo.jpg")



if __name__ == '__main__':

    app.run('127.0.0.1',port=5002,threaded=True)
