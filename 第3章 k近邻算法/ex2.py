### Loaidng packages
from skimage import io # 图像输入输出
from skimage.color import rgb2lab, lab2rgb # 图像通道转换
from sklearn.neighbors import KNeighborsRegressor # KNN 回归器
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
fig = plt.figure(figsize=(16, 5))
dir_path = 'style_transfer'
data_dir = os.path.join(dir_path,'vangogh')
# print(dir_path)
fig = plt.figure(figsize=(15,5))
print(os.listdir(data_dir))
for i, file in enumerate(np.sort(os.listdir(data_dir))[:3]):
    img = io.imread(os.path.join(data_dir,file))
    ax = fig.add_subplot(1,3,i+1)
    ax.imshow(img)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title(file)
plt.show()

## block_size 为向外扩展的层数
block_size = 1
def read_style_image(file_name,/,*,size = block_size):
    img = io.imread(file_name)
    plt.figure()
    plt.imshow(img)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title(file_name)
    plt.show()
    img = rgb2lab(img)
    w,h = img.shape[:2]
    X = []
    Y = []
    for i in range(size, w - size):
        for j in range(size, h - size):
            X.append(img[i - size : i + size + 1, j - size : j + size + 1, 0].flatten())
            Y.append(img[i, j, 1:])
    return X, Y
X, Y = read_style_image(os.path.join(dir_path,'style.jpg'))
knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
knn.fit(X,Y)

def build(img,/,*,size = block_size):
    fig = plt.figure()
    plt.imshow(img)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    img = rgb2lab(img)
    w, h = img.shape[:2]
    photo = np.zeros(shape=[w, h, 3])
    print('Constructing window...')
    X = []
    for x in range(size, w - size):
        for y in range(size, h - size):
            window = img[x - size : x + size + 1, y - size : y + size + 1, 0].flatten()
            X.append(window)
    X = np.array(X)
    print('Predicting...')
    pred_ab = knn.predict(X).reshape(w - 2*size, h - 2*size, - 1)
    photo[:, :, 0] = img[:, :, 0]
    photo[size : w - size, size: h - size, 1:] = pred_ab
    photo = photo[size : w - size, size: h - size, :] 
    return photo

content = io.imread(os.path.join(dir_path,'input.jpg'))
photo = build(content)
photo = lab2rgb(photo)
plt.figure()
plt.imshow(photo)
plt.show()



