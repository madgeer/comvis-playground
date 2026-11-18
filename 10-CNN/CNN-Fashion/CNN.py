import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.datasets import fashion_mnist

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
train_x = train_x.astype('float32') / 255.
test_x = test_x.astype('float32') / 255.
train_x = np.reshape(train_x, (len(train_x), 28, 28, 1))
test_x = np.reshape(test_x, (len(test_x), 28, 28, 1))
train_y = to_categorical( train_y )
test_y = to_categorical( test_y )

inputs = Input(shape=(28, 28, 1))
conv_layer = ZeroPadding2D(padding=(2,2))(inputs)
conv_layer = Conv2D(16, (5, 5), strides=(3,3), activation='relu')(conv_layer)
conv_layer = MaxPooling2D((2, 2))(conv_layer)
conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='relu')(conv_layer)
conv_layer = ZeroPadding2D(padding=(1,1))(conv_layer)
conv_layer = Conv2D(64, (3, 3), strides=(1,1), activation='relu')(conv_layer)

flatten = Flatten()(conv_layer)

fc_layer = Dense(256, activation='relu')(flatten)
fc_layer = Dense(64, activation='relu')(fc_layer)
outputs = Dense(10, activation='softmax')(fc_layer)
model = Model(inputs=inputs, outputs=outputs)
adam = Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy',
metrics=['accuracy'])

callbacks = TensorBoard(log_dir='./Graph')
model.fit(train_x, train_y, batch_size=256, epochs=100, verbose=1,
validation_data=(test_x, test_y), callbacks=[callbacks])
model.save_weights('weights.weights.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Ambil 5 gambar test
sample_x = test_x[:5]
sample_y = test_y[:5]

# Prediksi
preds = model.predict(sample_x, verbose=0)

for i in range(5):
    pred_label = class_names[np.argmax(preds[i])]
    true_label = class_names[np.argmax(sample_y[i])]
    print(f"Gambar {i}: Prediksi = {pred_label}, Asli = {true_label}")
    
    plt.imshow(sample_x[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {pred_label} | Asli: {true_label}")
    plt.axis('off')
    plt.show()