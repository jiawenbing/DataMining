# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:34:57 2019
KNN算法
@author: 贾文兵
"""
import numpy as np
from numpy import *

import matplotlib
import matplotlib.pyplot as plt

#解决可视化显示时的中文乱码问题
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname = 'C:\Windows\Fonts\simsun.ttc')

#创建数据源，返回数据集和类标签
def create_dataset():
    datasets = array([[8,4,2],[7,1,1],[1,4,4],[3,0,5]])
    labels = ['非常热','非常热','一般热','一般热']
    return datasets,labels

#可视化分析数据
def analyze_data_plot(x,y):
    fig = plt.figure()
    #将画布划分为一行一列一块
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    plt.show()
    #设置散点图标题和横纵坐标
    plt.title("游客冷热感知散点图", fontsize = 25, fontname = '宋体', fontproperties = myfont)
    plt.xlabel("天热吃冰激凌数目", fontsize = 15, fontname = '宋体', fontproperties = myfont)
    plt.ylabel("天热喝水数目", fontsize = 15, fontname = '宋体', fontproperties = myfont)
    #保存截图
    plt.savefig('datasets_plot.png', bbox_inches='tight')
    
#KNN分类器
import operator
def knn_Classifier(newV, datasets, labels, k):
    #1.获取待分类的数据
    #2.获取样本库的数据
    #3.选择K值
    #4.计算待分类的数据与样本库中每个数据的距离
    sqrtDist = euclideanDistance3(newV, datasets)
    #5.对距离进行从小到大排序,axis=0按列进行相应运算
    sortDistIndexs = sqrtDist.argsort(axis = 0)
    #print(sortDistIndexs)
    #6.针对K个点，统计各个类别的数量
    classCount = {}
    for i in range(k):
        #根据距离排序索引值找到类标签
        votelabel = labels[sortDistIndexs[i]]
        #print(sortDistIndexs[i],votelabel)
        #返回指定键的值，如果值不在字典中返回default值
        #统计类标签的键值对
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    #print(classCount)
    #7.确定待分类数据所属类别
    #对字典中的value值进行降序排序（operator.itemgetter(1)   为1时，表示按值进行排序；为0时，表示按键进行排序）
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    #print(newV,'KNN预测结果为：', sortedClassCount[0][0])
    return sortedClassCount[0][0]
#欧氏距离计算一
def euclideanDistance1(x1, y1, x2, y2):
    d = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    return d

#欧氏距离计算优化二
#instance为一个向量，length为向量的维
def euclideanDistance2(instance1, instance2, length):
    d = 0
    for x in range(length):
        d += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(d)

#欧氏距离计算优化三
def euclideanDistance3(newV, datasets):
    #1.获取数据集的条数和维数
    rowsize,colsize = datasets.shape
    #2.各特征向量间作差
    diffMat = tile(newV,(rowsize,1)) - datasets
    #print(diffMat)
    #3.对差值进行平方
    sqDiffMat = diffMat**2
    #print(sqDiffMat)
    #4.对差值平方和进行开方,axis=1表示按行进行运算
    sqrtDiffMat = sqDiffMat.sum(axis = 1)**0.5
    #print(sqrtDiffMat)
    return sqrtDiffMat

if __name__ == '__main__':
    #创建数据集和类标签
    datasets,labels = create_dataset()
    #print('数据集：\n',datasets,'\n标签:\n',labels)
    
    #可视化分析数据
    #analyze_data_plot(datasets[:,0],datasets[:,1])
    
    #单实例KNN分类器
    newV = [2,4,0]
    predict = knn_Classifier(newV, datasets, labels, 3)
    print(newV,'预测结果为：', predict)
    #多实例KNN分类器
    vecs = array([[2,4,4],[3,0,0],[5,7,2]])
    for vec in vecs:
        res = knn_Classifier(vec, datasets, labels, 3)
        print(vec,'预测结果为：', predict)
    #当要计算待分类样本与一个向量的距离，且向量的维数不多
    #euclideanDistance2([3, 2], [3, 1], 2)
    #当要计算待分类样本与一组数据集中所有的向量之间的距离时
    #euclideanDistance3([2,4,4], datasets)
    
    
