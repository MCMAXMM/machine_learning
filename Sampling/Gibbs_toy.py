import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
#为了方便操作，我们定义如下参数：
#建立个字典吧
#下面给的是标准正太分布，你可以修改参数
u1=0
u2=0
sigma_11=1
sigma_12=0
sigma_21=0
sigma_22=1
sample=[]
u=np.array([u1,u2],dtype=np.float32)
sigma=np.array([[sigma_11,sigma_12],[sigma_21,sigma_22]],dtype=np.float32)
def x1_given_x2(x2):
    u_cond=u1+sigma_12/sigma_22*(x2-u2)
    sigma_cond=sigma_11-sigma_12/sigma_22*sigma_21
    return stats.norm.rvs(loc=u_cond,scale=sigma_cond,size=1)
def x2_given_x1(x1):
    u_cond = u2 + sigma_12 / sigma_11 * (x1 - u1)
    sigma_cond = sigma_22 - sigma_12 / sigma_11 * sigma_12
    return stats.norm.rvs(loc=u_cond, scale=sigma_cond, size=1)

def Gibbs_Sampling(initial,size):
    sample.append(initial)
    for i in range(size):
        if i%2==0:
            x1=x1_given_x2(sample[-1][1])
            sample.append([x1,sample[-1][1]])
            continue
        else:
            x2=x2_given_x1(sample[-1][0])
            sample.append([sample[-1][0],x2])
    return sample
#初始样本点，我是自己定义的，你可以随机初始化
initial=np.array([5,8],dtype=np.float32)

b=Gibbs_Sampling(initial,151)
b=np.array(b)
for i in range(len(b)):
    plt.cla()
    plt.plot(b[0:i+1,0],b[0:i+1,1])
    plt.pause(0.03)#停顿时间，你可以调整
fig2=plt.figure()
ax=fig2.gca()
ax.scatter(b[:,0],b[:,1])
plt.show()





