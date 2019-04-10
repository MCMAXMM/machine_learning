#Multiple Dimensional Scaling,MDS,多维尺度缩放
#数据是按照行来排列的
#尽量让样本数目大于数据的维度
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.spatial.distance import pdist,squareform
class MDS():
    def __init__(self,dim):
        self.dim=dim
    def fit(self,X):
        self.X=X
        return self
    def predict(self):
        sample_number=self.X.shape[0]#样本数量
        #距离矩阵
        dist_2_matrix=squareform(pdist(self.X,"euclidean"))**2
        dist_i=np.mean(dist_2_matrix,axis=1).reshape((1,sample_number))
        dist_j=np.mean(dist_2_matrix,axis=0).reshape((sample_number,1))
        dist_ij=np.mean(dist_j)
        b=dist_2_matrix-dist_i-dist_j+dist_ij
        b=b*-0.5
        eigvalue,eigvector=np.linalg.eig(b)
        eig_diag=np.diag(eigvalue)
        Z_latent=eigvector[:,:self.dim].dot(eig_diag[:self.dim,:self.dim])
        return Z_latent
#下面为测试数据每一对都是在某个坐标轴上差了1，所以投射到二维空间距离还是比较近的
x=np.array([[2,2,2],[2,2,3],[9,4,9],[10,4,9],[3,45,23],[3,44,23]])
mds=MDS(2)
mds.fit(x)
b=mds.predict()
print(b)
# plt.scatter(x[:,0],x[:,1],c="green")
plt.scatter(b[:,0],b[:,1],c="red")
plt.show()






