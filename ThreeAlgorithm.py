
# coding: utf-8

# In[189]:




# In[521]:

from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import pandas as pd

#-----------批量梯度下降法----------------------

#alpha为步长,epsilon为终止迭代的最小距离,iteration为最大迭代次数.
def  BGD(x,y,theta,alpha,epsilon,iteration):
    for i in range(iteration):
        #拟合函数
        h = np.dot(x,theta)
        #梯度
        grad =np.dot( x.T,(h-y))
        #下降的距离
        dis = alpha*grad
        max_dis = abs(dis).max()
        
        if max_dis < epsilon : 
            break
        else:
            theta = theta - dis
    return theta

#------预测函数-------------------------------
def predict(x,theta):
    y = np.dot(x,theta)
    return y

#导入数据
boston = load_boston()
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
            
X = bos.drop('PRICE',axis = 1)
target = boston.target
y = np.reshape(target,(len(target),1))
            
#分割数据
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)
#初始化参数
theta = np.zeros((13,1))

#训练
theta = BGD(X_tr,y_tr,theta,1,0.1,30)
#预测
pred = predict(X_te,theta)
#精度
mse = np.mean((y_te-pred)**2)

#-------线性回归对比------------------------------------------
lm = LinearRegression()
lm.fit(X_tr,y_tr)

print('-------------------梯度下降法对比--------------------------')
print( '梯度下降法估计的theta:' ,theta)
print('部分测试集预测标签:',pred[10:15])
print('梯度下降法的MSE:',mse )
print('-------------------线性回归对比-----------------------------')
print('线性回归算法w值：', lm.coef_)
print('线性回归算法b值: ', lm.intercept_)
print('线性回归算法的MSE:',np.mean((y_te - lm.predict(X_te)) ** 2))

#----------牛顿法-------------------------------------------------

def  Newton(x,y,theta,alpha,epsilon,iteration):
    for i in range(iteration):
        #拟合函数
        h = np.dot(x,theta)
        #梯度
        grad =np.dot( x.T,(h-y))
        #hessian矩阵
        hessian = np.dot(x.T,x)
        
        p = np.dot(np.linalg.inv(hessian),grad)
        
        #距离
        dis = alpha*p
        max_dis = abs(dis).max()
        
        if max_dis < epsilon : 
            break
        else:
            theta = theta - dis
    return theta
#-----------预测函数----------
def predict(x,theta):
    y = np.dot(x,theta)
    return y

#训练
theta = Newton(X_tr,y_tr,theta,1,0.1,30)
#预测
pred = predict(X_te,theta)
#精度
mse = np.mean((y_te-pred)**2)

print('-------------------牛顿法对比-------------------------------')
print( '牛顿法估计的theta:' ,theta)
print('部分测试集预测标签:',pred[10:15])
print('牛顿法的MSE:',mse )


# In[520]:
#--------------------遗传算法----------------------
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

import os as o

print('---------------利用遗传算法求函数最大值-------------------------------------')

# 初始化
DNA_size = 6      #染色体维度
people_size = 100  #总群大小
cross_rate = 0.6     #基因交叉概率
mutate_rate = 0.1  #基因变异概率
itera = 10         #最大迭代次数

#解码,转成十进制数
def translate(pop): 
    return pop.dot(2 ** np.arange(DNA_size)[::-1]) / float(2**DNA_size-1) * 5
#定义适应函数
def fit(x):
    fit = np.sin(10*x) +2
    #fit = x**3 +8
    return fit
print('目标函数为:sin(10*x)+2')
print('目标:求解目标函数的最大值')

#选择: 随机普遍选择法
def select(pop):
    p = fit(translate(pop))/sum( fit(translate(pop)))
    idx = np.random.choice(np.arange(people_size),size = people_size,replace = True,p=p)
    idx_list = np.ndarray.tolist(idx)
    idx_remove = list(set(idx_list))   #去重
    return pop[idx_remove]
                           
                           
#交叉: 采用单点交叉,由'好'的样本产生新的子样本.

def cross(pop):
    parent = select(pop)
    parent_copy = parent.copy()
    parent_size = parent.shape[0]
    child_size = people_size - parent_size
    
    
    child1 = [ ]
    child2 = [ ]
    for i in range(child_size ):
        idx = np.random.choice(np.arange(parent_size),size = 2)
        a1 = parent[idx  [0] ] [3:6]
        b1 = parent[idx  [1] ] [0:3]
        if np.random.rand( ) < cross_rate:
            c1 = np.append(a1,b1)
        else:
            c1 = parent[idx [0]]
        
        
        child1.append(c1)
        
            
    return child1 ,parent_copy
    
    
    

#变异: 产生的子样本可能发生基因变异                     
def mutate(pop):
    child_new = cross(pop)[0]
    child_new_copy = child_new.copy()
    parent = cross(pop)[1]
    child_size = len(child_new)
    mutate_child = [ ]
    for i in range(child_size):
        if np.random.rand( ) < mutate_rate:
            
            idx = np.random.choice(np.arange(DNA_size),size = 1)
            child_new[i][idx] = 1 if child_new[i][idx] ==0 else 0
        else :
            child_new[i] == child_new_copy[i]
        mutate_child.append(child_new[i])
        mutate = np.array(mutate_child)
    return mutate,parent

# 将经过交叉,变异等产生的新子样本,和原先选出的"好的"样本组合成新的总群
def new_people(pop):
    new_people = np.concatenate((mutate(pop)[0],mutate(pop)[1]),axis =0)
    return new_people


#--------------------------------------------------------

#随机生成二进制串
pop = np.random.randint(2, size=(people_size, DNA_size))

#迭代 , 不断用好的子样本代替原来的坏样本:
for i in range(itera):
    new_people1 = new_people(pop)

tran = translate(new_people1)
fit = fit(tran)
root = np.max(fit)

print('遗传算法所求的解为:',root)



