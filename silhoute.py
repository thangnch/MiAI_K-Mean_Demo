import cv2
from sklearn.cluster import KMeans
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = cv2.imread("data/cars/364 (20).jpg")
c_data_path = "data/cars"

def get_feature(img):
    intensity = img.sum(axis=1)
    intensity = intensity.sum(axis=0) / (255 * img.shape[0] * img.shape[1])
    return intensity



def load_data(data_path=c_data_path):

    try:
        with open('data.pickle', 'rb') as handle:
            X = pickle.load(handle)
        with open('label.pickle', 'rb') as handle:
            L = pickle.load(handle)
        return X,L
    except:
        X = []
        L = []
        for file in os.listdir(data_path):
            print(file)
            c_x = get_feature(cv2.imread(os.path.join(data_path, file)))
            X.append(c_x)
            L.append(file)

        X = np.array(X)
        L = np.array(L)
        with open('data.pickle', 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('label.pickle', 'wb') as handle:
            pickle.dump(L, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return X,L


# Print the shape of the imag
X,L = load_data()
print(X.shape)

from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(2, 2, figsize=(15,8))
for k in [2, 3, 4, 5]:

    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(k, 2)

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X)

visualizer.show()

#
# kmeans = KMeans(n_clusters=4).fit(X)
# for i in range(len(kmeans.labels_)):
#     print(kmeans.labels_[i]," - ", L[i])
# print(kmeans.cluster_centers_)

#
# n_row = 6
# n_col=6
# for i in range(4):
#     _, axs = plt.subplots(n_row, n_col, figsize=(7, 7))
#     axs = axs.flatten()
#     for img, ax in zip(L[ kmeans.labels_ == i][:36], axs):
#         ax.imshow(mpimg.imread(os.path.join(c_data_path,img)))
#     plt.tight_layout()
#     plt.show()