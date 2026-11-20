import numpy as np
import tensorflow as tf
import random
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import TrainingPlot as tp
import MyUtils as mut

# 1. Supaya Repeatable
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# 2. Load Data & Konversi ke NumPy Array
print("Load Data")
lokasi_dataset = "dataset"
(imdata, imlabel) = mut.load(lokasi_dataset, 0, (20, 20))

# Diubah ke numpy array terlebih dahulu agar tidak error saat di-index atau di-rescale
X = np.array(imdata, dtype="float32") / 255.0
imlabel = np.array(imlabel)

# 3. Label Encoding & Binarization
print("Label Encoding")
le = LabelEncoder()
Ye = le.fit_transform(imlabel)

binarizer = LabelBinarizer()
Ye_b = binarizer.fit_transform(Ye)

# 4. Bagi Dataset (Cukup 1 Kali Split)
X_train, X_test, Y_train_b, Y_test_b, Y_train, Y_test = model_selection.train_test_split(
    X, Ye_b, Ye, test_size=0.2, shuffle=True, stratify=Ye, random_state=SEED
)

# 5. Buat Model Jaringan (Functional API)
inputs = Input(shape=(20, 20))
flatten = Flatten()(inputs)
h_layer = Dense(64, activation="relu")(flatten)
h_layer2 = Dense(64, activation="relu")(h_layer)
outputs = Dense(len(binarizer.classes_), activation="softmax")(h_layer2)
model = Model(inputs=inputs, outputs=outputs)

# Optimizer & Compile
adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

# 6. Proses Training
custom_plot = tp.TrainingPlot()

# Perbaikan: Menghilangkan spasi pada 'val_accuracy'
filepath = 'weights-improv-{epoch:02d}-{val_accuracy:.2f}.keras'
checkpoint = ModelCheckpoint(
    filepath, 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

H = model.fit(
    X_train, Y_train_b,
    validation_split=0.25,
    epochs=10,
    batch_size=32,
    callbacks=[custom_plot, checkpoint]
)

# 7. Proses Testing & Evaluasi
output = model.predict(X_test)
hasil_prediksi = binarizer.inverse_transform(output)

print('Hasil:')
print(metrics.classification_report(Y_test, hasil_prediksi))

print('Akurasi')
print(metrics.accuracy_score(Y_test, hasil_prediksi))

print('Precision')
print(metrics.precision_score(Y_test, hasil_prediksi, average='macro'))

print('Recall')
print(metrics.recall_score(Y_test, hasil_prediksi, average='macro'))

print('F1-Score')
print(metrics.f1_score(Y_test, hasil_prediksi, average='macro'))

print('Confusion Matrix:')
print(metrics.confusion_matrix(Y_test, hasil_prediksi))