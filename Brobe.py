import matplotlib.pyplot
import numpy



file = open('C:/Users/Kupyrev/Desktop/mnist_dataset/mnist_train_100.csv', 'r')
file_list = file.readlines()
file.close()

all_values = file_list[2].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()