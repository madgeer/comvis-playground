import numpy as np
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
#Kelas callback untuk visualiasi
class TrainingPlot(keras.callbacks.Callback):
    # Fungsi ini dipanggil saat training dimulai
    def on_train_begin(self, logs={}):
        # lakukan inisialisasi
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        plt.ion()
    
    #Fungsi ini dipanggil setiap akhir epoch
    def on_epoch_end(self, epoch, logs={}):
        # Ambil data dari parameter
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        # pastikan vis dilakukan ketika min terdapat 2 data
        if len(self.losses) > 1:
            N = np.arange(0, len(self.losses))
            plt.plot(N, self.losses,'r-', label = "train_loss")
            plt.plot(N, self.acc,'b-', label = "train_acc")
            plt.plot(N, self.val_losses,'g-', label = "val_loss")
            plt.plot(N, self.val_acc,'y-', label = "val_acc")
            plt.title("Loss dan Accuracy [Epch {}]".format(epoch))
            if(len(self.losses)==2):
                plt.xlabel("Epoch #")
                plt.ylabel("Loss/Accuracy")
                plt.legend()
        plt.pause(0.00000001)
        # Simpan grafik, pastikan direktori tersedia
        plt.savefig('output/Epoch-{}.png'.format(epoch))