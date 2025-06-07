#Imports
    import os
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#More imports
import tensorflow as tf
from github import Github
from matplotlib import pyplot as plt
import numpy as np
import cv2
import requests
from PIL import Image
import imghdr

#Limits memory used by GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Creating data directories
parent_dir = os.getcwd()
data_dir = 'data'
data_directory = os.path.join(parent_dir, data_dir)
os.mkdir(data_directory)
directory0 = "fireAlarmPictures"
fireAlarmDirectory = os.path.join(data_directory, directory0)
os.mkdir(fireAlarmDirectory)
directory1 = "firstAidPictures"
firstAidDirectory = os.path.join(data_directory, directory1)
os.mkdir(firstAidDirectory)

#URL builder
def get_url_paths(directory, urlTag):
    parent = 'https://raw.githubusercontent.com/TristanFlores7/imageClassificationDataset/main/' + directory + urlTag
    return parent

#Gathering image dataset from the web (GitHub)
g = Github()
repos = g.search_repositories(query="owner:TristanFlores7")

for repo in repos:
    if repo.full_name == "TristanFlores7/imageClassificationDataset":
        #Grabbing fire alarm pictures
        content0 = repo.get_contents(path="fireAlarmPictures")
        for content_file0 in content0:
            print(get_url_paths("fireAlarmPictures/", content_file0.name))
            resp = requests.get(url=get_url_paths("fireAlarmPictures/", content_file0.name), stream=True).raw
            file_path0 = os.path.join(fireAlarmDirectory, content_file0.name)
            print(file_path0)
            with Image.open(io.BytesIO(resp.read())) as img0:
                img0.save(file_path0)
        #Grabbing first-aid kit pictures
        content1 = repo.get_contents(path="firstAidPictures")
        for content_file1 in content1:
            print(get_url_paths("firstAidPictures/", content_file1.name))
            resp = requests.get(url=get_url_paths("firstAidPictures/", content_file1.name), stream=True).raw
            file_path1 = os.path.join(firstAidDirectory, content_file1.name)
            print(file_path1)
            with Image.open(io.BytesIO(resp.read())) as img1:
                img1.save(file_path1)

#Image processing
print(os.listdir(data_dir))
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

#Removes sketchy image files
for image_class in os.listdir((data_dir)):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in an ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            #os.remove(image_path)

#Creating data pipeline
print(tf.config.list_physical_devices('GPU'))
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#Scaling data
data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()

#Establishing training/testing size
print(len(data))
train_size = 5
valid_size = 2
test_size = 1

train = data.take(train_size)
valid = data.skip(train_size).take(valid_size)
test = data.skip(train_size + valid_size).take(test_size)

#Building Deep Learning Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
print(model.summary())

#Training the model
#os.mkdir('logs')
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=valid, callbacks=[tensorboard_callback])
print(hist.history)

#Plot performance
fig0 = plt.figure()
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='yellow', label='validation loss')
fig0.suptitle('Loss', fontsize=24)
plt.legend(loc="upper left")
plt.show()

fig1 = plt.figure()
plt.plot(hist.history['accuracy'], color='blue', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='green', label='validation accuracy')
fig1.suptitle('Accuracy', fontsize=24)
plt.legend(loc="upper left")
plt.show()

#Model testing
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'\nPrecision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

#Single image test example:

#First grabbbing test image from url
testIm = requests.get(url='https://img.fruugo.com/product/0/08/700706080_max.jpg', stream=True).raw
test_path = os.path.join(data_dir, 'testExample.jpeg')
with Image.open(io.BytesIO(testIm.read())) as img1:
    img1.save(test_path)
testImg = cv2.imread(test_path)
plt.imshow(cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB))
plt.show()

#Resizing and formating the image
resize = tf.image.resize(testImg, (256,256))
np.expand_dims(resize, 0)
yhat = model.predict(np.expand_dims(resize/255, 0))
print('\nPrediction: ', yhat)
if yhat > 0.5:
    print('Model prediction is more than 0.5, so predicted class is first-aid kit')
else:
    print('Model prediction is less than 0.5, so predicted class is fire alarm')
