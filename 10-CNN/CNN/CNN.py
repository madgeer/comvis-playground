import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import MyUtils as mut

print("Load Data")
lokasi_dataset = "dataset"
(imdata, imlabel) = mut.load(lokasi_dataset, 0, (20, 20))

train_x = np.array(imdata, dtype="float32") / 255.
train_x = np.expand_dims(train_x, axis=-1)

le = LabelEncoder()
Ye = le.fit_transform(imlabel)

binarizer = LabelBinarizer()
Ye_b = binarizer.fit_transform(Ye)

X_train, X_test, Y_train_b, Y_test_b, Y_train, Y_test = model_selection.train_test_split(
    train_x, Ye_b, Ye, test_size=0.2, shuffle=True, stratify=Ye, random_state=42
)
 
inputs = Input(shape=(20, 20, 1))
conv_layer = ZeroPadding2D(padding=(2,2))(inputs)
conv_layer = Conv2D(16, (5, 5), strides=(3,3), activation='relu')(conv_layer)
conv_layer = MaxPooling2D((2, 2))(conv_layer)
conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='relu')(conv_layer)
conv_layer = ZeroPadding2D(padding=(1,1))(conv_layer)
conv_layer = Conv2D(64, (3, 3), strides=(1,1), activation='relu')(conv_layer)

flatten = Flatten()(conv_layer)

fc_layer = Dense(256, activation='relu')(flatten)
fc_layer = Dense(64, activation='relu')(fc_layer)
outputs = Dense(len(le.classes_), activation='softmax')(fc_layer)
model = Model(inputs=inputs, outputs=outputs)
adam = Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy',
metrics=['accuracy'])

callbacks = TensorBoard(log_dir='./Graph')
model.fit(X_train, Y_train_b, batch_size=256, epochs=100, verbose=1,
validation_data=(X_test, Y_test_b), callbacks=[callbacks])
model.save_weights('weights.weights.h5')

output = model.predict(X_test)
y_pred_classes = np.argmax(output, axis=1)
y_pred_labels = le.inverse_transform(y_pred_classes)
y_test_labels = le.inverse_transform(Y_test)

print('Hasil:')
print(metrics.classification_report(y_test_labels, y_pred_labels))

print('Akurasi')
print(metrics.accuracy_score(y_test_labels, y_pred_labels))

print('Precision')
print(metrics.precision_score(y_test_labels, y_pred_labels, average='macro'))

print('Recall')
print(metrics.recall_score(y_test_labels, y_pred_labels, average='macro'))

print('F1-Score')
print(metrics.f1_score(y_test_labels, y_pred_labels, average='macro'))

print('Confusion Matrix:')
print(metrics.confusion_matrix(y_test_labels, y_pred_labels))

class_names = le.classes_

# Ambil 5 gambar test
sample_x = X_test[:5]
sample_y = Y_test_b[:5]

# Prediksi
preds = model.predict(sample_x, verbose=0)

for i in range(5):
    pred_label = class_names[np.argmax(preds[i])]
    true_label = class_names[np.argmax(sample_y[i])]
    print(f"Gambar {i}: Prediksi = {pred_label}, Asli = {true_label}")
    
    plt.imshow(sample_x[i].reshape(64, 64), cmap='gray')
    plt.title(f"Pred: {pred_label} | Asli: {true_label}")
    plt.axis('off')
    plt.show()
