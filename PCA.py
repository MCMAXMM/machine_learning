import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import svd_flip
class PCA:
    def __init__(self,dim,Decomposition="eig"):
        self.Decom_method=Decomposition
        self.dim=dim
    def fit(self,X):
        self.input=X
        return self
    def transform(self):
        original_dim=self.input.shape[1]
        #样本中心化
        sample_mean=np.mean(self.input,axis=0).reshape((1,original_dim))
        center=self.input-sample_mean
        if self.Decom_method=="eig":
            Covariance_mat=center.T.dot(center)
            eigvalue,eigvector=np.linalg.eig(Covariance_mat)
            project_matrix=eigvector[:,:self.dim]
            project_data=center.dot(project_matrix)
        elif self.Decom_method=="svd":
            U,D,V=np.linalg.svd(center)
            U=-U#强制翻转符号，保证输出一致，sklearn中使用的svd_flip来做的，不然老是和特征分解的差一个符号
            #UDV=(-U)D(-V)
            D=np.diag(D)
            project_data=U[:,:self.dim].dot(D[:self.dim,:self.dim])
        return project_data
x=np.array([[2,2,2],[2,2,3],[9,4,9],[10,4,9],[3,45,23],[3,44,23]])
pca=PCA(dim=2,Decomposition="svd")
pca.fit(x)
data=pca.transform()
print(data)
plt.scatter(data[:,0],data[:,1])
plt.show()
