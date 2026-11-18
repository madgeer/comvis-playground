import os
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
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

# Buat direktori output untuk TrainingPlot
os.makedirs('output', exist_ok=True)

# 2. Load Data & Konversi ke NumPy Array
print("Load Data")
lokasi_dataset = "dataset"
(imdata, imlabel) = mut.load(lokasi_dataset, 0, (64, 64))

X = np.array(imdata, dtype="float32") / 255.0
# Tambah channel dimension untuk CNN (grayscale): (N, 64, 64) -> (N, 64, 64, 1)
X = np.expand_dims(X, axis=-1)
imlabel = np.array(imlabel)

# 3. Label Encoding & Binarization
print("Label Encoding")
le = LabelEncoder()
Ye = le.fit_transform(imlabel)

binarizer = LabelBinarizer()
Ye_b = binarizer.fit_transform(Ye)

# 4. Bagi Dataset (Train / Test)
X_train, X_test, Y_train_b, Y_test_b, Y_train, Y_test = model_selection.train_test_split(
    X, Ye_b, Ye, test_size=0.2, shuffle=True, stratify=Ye, random_state=SEED
)

# 5. Buat Model CNN (Functional API)
inputs = Input(shape=(64, 64, 1))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(binarizer.classes_), activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

# Optimizer & Compile
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

# 6. Proses Training
custom_plot = tp.TrainingPlot()

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
    validation_data=(X_test, Y_test_b),
    epochs=50,
    batch_size=16,
    callbacks=[custom_plot, checkpoint]
)

# 7. Proses Testing & Evaluasi
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

# Tampilkan 5 contoh prediksi
class_names = le.classes_
sample_x = X_test[:5]
sample_y_int = Y_test[:5]
preds = model.predict(sample_x, verbose=0)

for i in range(5):
    pred_label = class_names[np.argmax(preds[i])]
    true_label = class_names[sample_y_int[i]]
    print(f"Gambar {i}: Prediksi = {pred_label}, Asli = {true_label}")

    plt.imshow(sample_x[i].reshape(64, 64), cmap='gray')
    plt.title(f"Pred: {pred_label} | Asli: {true_label}")
    plt.axis('off')
    plt.show()
