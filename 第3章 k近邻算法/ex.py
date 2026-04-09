import numpy as np
import matplotlib.pyplot as plt
import os

m_x = np.loadtxt('mnist_x', delimiter = ' ')
m_y  = np.loadtxt('mnist_y')
### x[0] 二维数组 这样切片是第一个样本
data = np.array(m_x[0]).reshape(28,28)
plt.figure()
plt.imshow(data,cmap = 'grey')
#plt.show()

# 将数据分为训练集和测试集
ratio = 0.8
#print(len(m_x)) 对于二维数组打印的是行数
split = int(len(m_x) * ratio)

# 打乱数据
np.random.seed(0)
idx = np.random.permutation(np.arange(len(m_x))) # arrange函数

m_x  = m_x[idx]# 选出这些行
m_y = m_y[idx]
x_train, y_train = m_x[:split], m_y[:split]
x_test, y_test = m_x[split:], m_y[split:]

### 建立KNN
#### 使用Euclidean norm
def distance(a,b):
	return np.sqrt(np.sum(np.square(b-a)))

### 建立KNN类
class KNN:
	
	def __init__(self,k,label_num):
		self.k = k
		self.label_num = label_num
	# 在类中保存数据
	def fit(self,x_train,y_train):
		self.x_train = x_train
		self.y_train = y_train
	# ***** 返回在x_train数据中最近的k个数据的位置 ——> list
	def get_knn_indices(self,x):
		dis = list(map(lambda a: distance(a,x), self.x_train))
		knn_indices = np.argsort(dis)
		##
		knn_indices = knn_indices[:self.k]
		return knn_indices

	def get_label(self,x):
		knn_indices = self.get_knn_indices(x)
		label_statistic = np.zeros(shape = [self.label_num])
		for index in knn_indices:
			label = int(self.y_train[index])
			label_statistic[label] += 1
		return np.argmax(label_statistic)


	def predict(self,x):
		predict_label = np.zeros(shape = len(x))
		for i,x in enumerate(x):
			predict_label[i] = self.get_label(x)
		return predict_label

for k in range(1,10):
	knn = KNN(k,label_num = 10)
	knn.fit(x_train,y_train)
	predict_label = knn.predict(x_test)
	
	accuray = np.mean(predict_label == y_test)
	print(f"当{k}近邻时，分类准确率是{accuray}")




		
			
		
		



