import numpy as np
import matplotlib.pyplot as plt
c=np.random.random(10)
from scipy import stats
#真实的概率分布
P_X=stats.multivariate_normal(mean=[2,3],cov=[[2,1],[1,2]]).pdf
#使用我们使用的均匀函数来作为状态转移函数
def uniform_sample(num):
  x_1=np.random.uniform(-8,8,num).reshape([num,1])
  x_2=np.random.uniform(-8,8,num).reshape((num,1))
  sample=np.concatenate((x_1,x_2),axis=1)
  return sample
  
def mtropolis(num):
  sample=list()
  #随机初始一个值，前期由于马尔科夫链还没有到达稳定状态，可以摒弃前期的样本
  x_t = [3, 5]
  #开始抽样
  while len(sample)<num:
    x_new=uniform_sample(1)
    #计算接受概率
    q=P_X(x_new)/P_X(x_t)
    u=np.random.uniform(0,1,1)
    if u<min(q,1):
      x_t=x_new
      sample.append(x_new)
  return sample
sam=mtropolis(1000)
sam=[a.reshape(2,) for a in sam]
x=[x[0] for x in sam]
y=[y[1] for y in sam]
plt.scatter(x,y)
plt.show()
