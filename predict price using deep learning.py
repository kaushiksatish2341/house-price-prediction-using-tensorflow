import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

a=pd.read_csv('123.csv')
dataset=a.values

X=dataset[:,:10]
y=dataset[:,10]

X_scale=scalar.fit_transform(X)

X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X_scale,y,test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('acc')>0.80):
            print('\nReached 80% accuracy so cancelling training!')
            self.model.stop_training=True

callbacks=myCallback()

model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),])
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model.fit(X_train, y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, y_val),callbacks=[callbacks])

print("Accuracy on training set:{:.3f}".format(model.evaluate(X_train,y_train)[1]))
print("Accuracy on Validation set:{:.3f}".format(model.evaluate(X_val,y_val)[1]))
print("Accuracy on test set:{:.3f}".format(model.evaluate(X_test,y_test)[1]))
