from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("train_image[0] : ")
print(train_images[0])

print("\n")

print("train_labels[0] : ")
print(train_labels[0])

print("shape of train_images")
print(train_images.shape)

print("shape of train_labels")
print(train_labels.shape)

print("\n")
print("\n")



import matplotlib.pyplot as plt

plt.gcf().set_size_inches(15, 4)
for i in range(5):
    ax = plt.subplot(1, 5, 1+i)
    ax.imshow(train_images[i], cmap= 'gray')
    ax.set_title('label = '+str(train_labels[i]), fontsize=18)
plt.show()



x_train = train_images.reshape((60000, 28 * 28))
x_train = x_train.astype('float32') / 255

x_test = test_images.reshape((10000, 28 * 28))
x_test = x_test.astype('float32') / 255

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print("before One-hot coding, label: ")
print(train_labels[0])
print("after One-hot coding, label: ")
print(y_train[0])

print("\n")

print('shape of train_labels :')
print(train_labels.shape)
print("\n")
print('shape of test_labels :')
print(test_labels.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(512, activation='relu', input_dim= 784))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['acc']) 

model.summary()

history = model.fit(x_train, y_train, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('accuracy to testing dataset : ', test_acc)

predict = model.predict(x_test)
predict.round(1)

predict = model.predict_classes(x_test)
print(predict)
print(test_labels)

predict = model.predict_classes(x_test)

plt.gcf().set_size_inches(15, 4)
for i in range(5):
	ax = plt.subplot(1, 5, 1+i)
	ax.imshow(test_images[i], cmap='binary')
	ax.set_title('label = '+str(test_labels[i]) +
				 '\npredi = '+str(predict[i]), fontsize=18)
	ax.set_xticks([]); ax.set_yticks([])
plt.show()
