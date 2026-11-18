import cv2
import numpy as np
from prometheus_client import Histogram
import MyUtils as mut
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn import metrics
from sklearn import model_selection

#load data
print("Load Data")
lokasi_dataset = "dataset"
(imdata,imlabel) = mut.load(lokasi_dataset,0)
X = imdata
#encode label
print("Label Encoding")
le = LabelEncoder()
Ye = le.fit_transform(imlabel)

#Ekstraksi Local Features menggunakan SIFT
print("Ekstraksi Local Features")
sift = cv2.SIFT_create()
local_feature_list = []
for img in X:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        local_feature_list.append(descriptors)

#Clustering untuk membuat codebook
print("Bangun Codebook")
k = 100 #jumlah vocab
all_words = np.vstack(local_feature_list)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(all_words)

#representasikan citra menjadi bovw
def get_bovw_hist(img, kmeans):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    histogram = np.zeros(len(kmeans.cluster_centers_))
    cluster_result = kmeans.predict(descriptors)
    for idx in cluster_result:
        histogram[idx] += 1
    return histogram
print("Ubah Representasi Citra menjadi Histogram")
representasi_X = [get_bovw_hist(img,kmeans) for img in X]

#bagi dataset
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(representasi_X, Ye, test_size=0.2,
 shuffle=True, stratify=Ye,
random_state=42)

#Lakukan Klasifikasi
#Buat model
scaler = StandardScaler()
svm = SVC(kernel='linear', probability=True)
pipeline = make_pipeline(scaler, svm)
#Training Model dengan data training
print('Training Model')
pipeline.fit(X_train, Y_train)
#Testing model
print('Testing Model')
hasil_prediksi = pipeline.predict(X_test)

#Akurasi
print('Akurasi')
print(metrics.accuracy_score(Y_test, hasil_prediksi))
#Precision
print('Precision')
print(metrics.precision_score(Y_test, hasil_prediksi, average='macro'))
#Recall
print('Recall')
print(metrics. recall_score(Y_test, hasil_prediksi , average='macro'))
#F1
print('F1-Score')
print(metrics. f1_score(Y_test, hasil_prediksi , average='macro'))
#Confusion Matrix
print(metrics.confusion_matrix(Y_test, hasil_prediksi))