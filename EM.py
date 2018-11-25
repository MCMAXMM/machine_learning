import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
np.random.seed(1234)
x_plot_1,x_plot_2=np.mgrid[-10:10:50j,-10:10:50j]
x_plot=np.stack((x_plot_1,x_plot_2),axis=2)
u=[[-5,-3],[3,-1],[-3,5]]#三个二元高斯的均值
sigma=[np.identity(2),np.diag([2,3]),np.array([[2,1],[1,2]])]#三个高斯的方差
#data1=np.random.multivariate_normal()
alpha=[0.1,0.5,0.4]
data_num=900
#上面是混合高斯的所有参数
#下面开始制造假的的数据
def fake_data_gen(u,sigma,alpha,number):
    norm_list=[]
    data_list=[]
    for ga_par in zip(u,sigma,alpha):
        norm=stats.multivariate_normal(ga_par[0],ga_par[1])
        norm_list.append(norm)
    for i in range(number):
        rand=np.random.uniform(0,1)
        if i%200==0:
            print("have generrate {} samples".format(i))
        if rand<=alpha[0]:
            data=norm_list[0].rvs(1)
        elif alpha[0]<rand<=(1-alpha[2]):
            data=norm_list[1].rvs(1)
        else:
            data=norm_list[2].rvs(1)
        data_list.append(data)
    return np.array(data_list)
data=fake_data_gen(u,sigma,alpha,data_num)
#画出假数据的散点图
def plot_mygraph(u,sigma):
    plt.cla()
    plt.scatter(data[:, 0], data[:, 1])
    for i in range(3):

        pdf_x = stats.multivariate_normal(mean=u[i], cov=sigma[i]).pdf(x_plot)
        plt.contour(x_plot_1,x_plot_2,pdf_x)


#初始化所有的参数，u，sigma,alpha
u=np.array([[1,1],[3,3],[6,6]])
sigma=np.array([np.identity(2),np.identity(2),np.identity(2)])
alpha=np.array([0.5,0.1,0.4])
#开始迭代
#是哪个分布的后验概率
def poserier_vec_sum(u,sigam,data,l):
    pdf_data_l=stats.multivariate_normal(u[l],sigma[l]).pdf(data)
    sum_data=0
    for i in range(3):
        pdf_data_i=stats.multivariate_normal(u[i],sigma[i]).pdf(data)
        sum_data+=pdf_data_i
    sum_data=pdf_data_l/sum_data
    return sum(sum_data)
def posterier(u,sigam,x_i,l):
    pdf_x_i=stats.multivariate_normal(u[l],sigma[l]).pdf(x_i)
    p_x_sum=sum([stats.multivariate_normal(u[i],sigma[i]).pdf(x_i) for i in range(3)])
    return  pdf_x_i/p_x_sum

def com_alpha(u,sigam,l,data,num):
    sum1=poserier_vec_sum(u,sigma,data,l)
    return sum1/(num)
def com_u(u,sigma,l,data):
    sum1 = poserier_vec_sum(u, sigma, data, l)
    sum2=0
    for x_i in (data):
        sum2+=x_i*posterier(u, sigma, x_i, l)
    return sum2/sum1
def com_sigma(new_u,u,sigma,l,data):
    sum2=0
    sum1 = poserier_vec_sum(u, sigma, data, l)
    for x_i in data:
        x_i_u=np.reshape(x_i,(1,2))-np.reshape(new_u[l],(1,2))
        x_i_u_T=np.reshape(x_i,(2,1))-np.reshape(new_u[l],(2,1))
        sum2+=np.matmul(x_i_u_T,x_i_u)*posterier(u,sigma,x_i,l)
    return sum2/sum1
def gussian_mixture_iter(u,sigma,alpha,data,iter_num):
    for i in range(iter_num):
        plot_mygraph(u, sigma)
        for l in range(3):
            alpha[l]=com_alpha(u,sigma,l,data,data_num)
        alpha = alpha / sum(alpha)
        new_u=[]
        for l in range(3):
            u_tmp=com_u(u,sigma,l,data)
            new_u.append(u_tmp)
        sigma_list=[]
        for l in range(3):
            sigma_tmp=com_sigma(new_u,u,sigma,l,data)
            sigma_list.append(sigma_tmp)
        u=new_u
        sigma=sigma_list
        alpha = alpha / sum(alpha)

        if i%1==0:
            print("第{}次迭代均值为：{}".format(i,u))
            print("第{}次迭代方差为:{}".format(i,sigma))
            print("第{}次迭代alpha为：{}".format(i, alpha))
        plt.show()
    return u,sigma,alpha
u,sigma,alpha=gussian_mixture_iter(u,sigma,alpha,data,100)
