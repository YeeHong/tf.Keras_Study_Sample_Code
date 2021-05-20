from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

model = load_model('MnistModel.h5')
(_,_), (test_images, test_labels) = mnist.load_data()
x_test = test_images.reshape((10000, 28 * 28))
x_test = x_test.astype('float32') / 255
y_test = to_categorical(test_labels)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('accuracy of test data : ', test_acc)
