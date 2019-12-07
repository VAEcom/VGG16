#coding:utf-8
import numpy as np
#Linux服务器没有GUI的情况下使用mayplotlib绘图，必须置于pyplot之前
#import matplotlib
#matplotlib.use("Agg")
import tensorflow as tf
import matplotlib.pyplot as plt

#下面三个是引用自定义模块
import vgg16
import utils
from Nclasses import labels

img_path=raw_input('Input the path and image name:')
img_ready=utils.load_image(img_path)#调用load_image()函数，对待测试的图像做一些>预处理操作
print "img_ready shape",tf.Session().run(tf.shape(img_ready))

#定义一个figure画图窗口，并指定窗口的名称，也可以设置窗口修的大小
fig=plt.figure(u"Yop-5 预测结果")

with tf.Session() as sess:
    #定义一个维度为[1,224,224,3],类型为float32的tensor展位符
    x=tf.placeholder(tf.float32,[1,224,224,3])
    vgg=vgg16.Vgg16()#类Vgg16实例化出vgg
    #调用类成员方法forward(),并传入待测试图像，这也就是网络前向传播过程
    vgg.forward(x)
    #将一个batch的数据喂入网络，得到网络的预测输出
    probablity=sess.run(vgg.prob,feed_dict={x:img_ready})
    #np.argsort函数返回预测值（probablity的数据结构[[各预测类别的概率值]])由小到
大的索引值
    #并取出测出概率最大的5个索引值
    top5=np.argsort(probability[0])[-1:-6:-1]
    print"top5:",top5
    #定义两个list---对应的概率值和实际标签
    values=[]
    bar_label=[]

    for n,i in enumerate(top5):#枚举上面5个索引值
        print "n:",n
        print "i:",i

        values.append(probability[0][i])#将索引值对应的预测概率值取出放入values
        bar_label.append(labels[i])#根据索引值取出实际的标签并放入bar_label
        print i,":",labels[i],"---",utils.percent(probability[0][i])#打印属于某>个类别的概率

    ax=fig.add_subplot(111)#将画布分为一行一列并将下图放入其中
    #bar()函数绘制柱状图，参数range(len(values))是柱子下标，values表示柱高的列表
（也就是5个预测概率值）
    #tick_label是每个柱子上显示的标签（实际对应的标签），width是柱子的宽度，fc是
柱子的颜色
    ax.bar(range(len(values)),values,tick_label=bar_label,width=0.5,fc='green')
    ax.set_ylabel('probability')#设置纵轴标签
    ax.set_title('Top-5')#添加标签
    for a,b in zip(range(len(values)),values):
        #在每个柱子的顶端添加对应的预测概率值，a,b表示坐标值，b+0.0005表示要把文
本信息放置在高于柱子顶端0.0005的位置
        #center是表示文本位于柱子顶端水平方向上的中间位置，bottom是将文本水平放>置在柱子顶端垂直方向上的底端位置，fontsize是字号
        ax.text(a,b+0.0005,utils.percent(b),ha='center',va='bottom',fontsize=7)
    plt.savefig('./result.jpg')#保存图片
    plt.show()#弹窗显示图像（linux服务器上将该句注销）
