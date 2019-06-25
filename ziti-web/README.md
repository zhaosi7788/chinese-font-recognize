# Chinese-font-recognition
Chinese font recognition

## 准备工作

- 从网上下载相对应的.ttf文件类型的字体文件，将其放入```dataset/fonts```文件夹下。
- 运行```dataset/generator.py```，程序会根据```fonts```文件夹下的字体文件对其中的汉字字体图片进行提取，并保存到对应的文件夹。
- 程序会根据“常用汉字大全.txt”中的内容对文字进行提取，如需改变提取内容，修改本文档即可。


## 训练

- 运行```train.py```进行模型参数的训练，修改其中的

```python
class_num = 9 
```
- 如果想改变迭代的次数，修改```data_pipline```函数中的```rang```参数即可：

```python
def data_pipline(batch_size):
    data_batch, annotation = read_data(batch_size)
    iterator = data_batch.make_initializable_iterator()
    inputs, outputs = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for _ in range(1500):
            data = sess.run([inputs, outputs])
            message.put(data)
    message.put(None)
```
- 模型会保存在```models```文件夹下，通过修改```saver.save```函数的参数可修改模型名称

```python
saver.save(sess, './models/vgg.ckpt')
```

## 测试
- 运行```demo.py```文件可以对训练好的模型进行测试。将```run```函数中的下列语句中的```“9”```修改为对应的分类数即可。

```python
outputs = vgg(example, 9, 1.0)
```

```python
        if index % 9  != label:
            error_texts.append((text, pred.tolist()))
            error += 1
```

- 如需进行模型测试结果的显示，需要将以下语句的注释删除即可，运行时间相应的会变长。

```python
# show_errors(error_texts, fonts)
```
- 测试结果会将判错图片进行显示，并将判别概率显示在图片上方。

## 展示平台
- 运行 ```web.py```文件即可启动字体识别系统，并显示其入口。端口地址可在以下语句中修改。

```python
app.run('127.0.0.1',port=5002,threaded=True)
```
- 以下语句用来进行模型的重载，需要将```output = vgg(example, 9, 1.0)```中的```“9”```修改为类别数量。

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 不使用GPU
inputs = tf.placeholder(tf.float32, shape=[None, None, 3])
example = tf.cast(tf.image.resize_images(inputs, [128, 128]), tf.uint8)
example = tf.image.per_image_standardization(example)
example = tf.expand_dims(example, 0)
output = vgg(example, 9, 1.0)
sess = tf.Session()
tf.train.Saver().restore(sess, 'models/vgg.ckpt')
print("模型重载成功")
```
- 前端页面为```template/upload```，可根据自身需要进行修改。
- 上传的图片存储在```static/uploads```中。
