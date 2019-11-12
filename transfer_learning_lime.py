import matplotlib.pyplot as plt 
import matplotlib.image  as mpimg
import tensorflow as tf 
import numpy as np 
import PIL.Image as Image
import os 
import keras
import tensorflow_hub as hub 
import math
from keras import layers 
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from skimage.io import imread
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

#setup some file exploration functions - credit: mannybernabe
def search_dir(type="normal", num=6):
  #Helper function to scan contents of directory
  counter=0
  for file in os.listdir(train_dir + type + "/"):
      if counter == num:
          break

      if file.endswith(".jpg"):
          print(file)
      counter += 1

def plot_images(type="normal", num=6):
  #Helper function to plot images
    counter=0
    plt.figure(figsize=(10, 8))  
    for file in os.listdir(train_dir + type):
        if file.endswith(".jpg"):
            if counter == num:
              break
            img = mpimg.imread(train_dir + type +"/"+file)
            plt.subplot(231+counter)
            plt.title(file.split('.')[0])
            plt.imshow(img)
            counter += 1
    plt.show()

#Setup the data directory
train_dir = "/media/sf_VirtualBox_Share/Training/"
valid_dir = "/media/sf_VirtualBox_Share/Validation/"
predict_dir = "/media/sf_VirtualBox_Share/Prediction2/"

#explore the directory
#search_dir(type = "Contemporary", num=10)
#plot_images(type = "Contemporary", num=6)

#define image parameters
IMAGE_SHAPE=(299,299)
BATCH_SIZE = 50
datagen = ImageDataGenerator(rescale=1./255) 

#define the pretrained headless model
headless_model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
                   trainable=False),  # Can be True, see below.
    ])
headless_model.build([None, 299, 299, 3])  # Batch input shape.


def save_bottleneck():   
# generate the training data  
    generator = datagen.flow_from_directory(
            train_dir,
            target_size=IMAGE_SHAPE,
            class_mode='categorical',
            shuffle=False
            )
    generator.batch_size = generator.classes.size

# check the training data
    for image_batch, label_batch in generator:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break      
# generate bottlenecks
    bottleneck_features_train = headless_model.predict(image_batch, batch_size = BATCH_SIZE, verbose = 1)

# save bottlenecks
    np.save(open('/media/sf_VirtualBox_Share/bottleneck_flat_features_train.npy', 'wb'), bottleneck_features_train)
    np.save(open('/media/sf_VirtualBox_Share/bottleneck_train_labels.npy', 'wb'), label_batch)
    
# generate validation data
    generator = datagen.flow_from_directory(
            valid_dir,
            target_size=IMAGE_SHAPE,
            class_mode='categorical',
            shuffle=False)
    generator.batch_size = generator.classes.size

# check validation data
    for image_batch, label_batch in generator:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break         

# generate bottlenecks
    bottleneck_features_valid = headless_model.predict(image_batch, batch_size = BATCH_SIZE, verbose = 1)

# save bottlenecks 
    np.save(open('/media/sf_VirtualBox_Share/bottleneck_flat_features_valid.npy', 'wb'), bottleneck_features_valid)
    np.save(open('/media/sf_VirtualBox_Share/bottleneck_valid_labels.npy', 'wb'), label_batch)



def train_top_model():
# load bottlenecks
    train_data = np.load(open('/media/sf_VirtualBox_Share/bottleneck_flat_features_train.npy', 'rb'))
    train_labels = np.load(open('/media/sf_VirtualBox_Share/bottleneck_train_labels.npy', 'rb'))
    validation_data = np.load(open('/media/sf_VirtualBox_Share/bottleneck_flat_features_valid.npy', 'rb'))
    validation_labels = np.load(open('/media/sf_VirtualBox_Share/bottleneck_valid_labels.npy', 'rb'))

# define the model
    top_model = keras.Sequential()
    top_model.add(keras.layers.Dense(1024, input_shape= train_data.shape[1:], activation='relu'))
    top_model.add(keras.layers.Dropout(0.5))
    top_model.add(keras.layers.Dense(7, activation='softmax'))
    top_model.summary()

# compile the model
    top_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    class CollectBatchStats(tf.keras.callbacks.Callback):
        def __init__(self):
            self.batch_losses = []
            self.batch_acc = []

        def on_train_batch_end(self, batch, logs=None):
            self.batch_losses.append(logs['loss'])
            self.batch_acc.append(logs['accuracy'])
            self.model.reset_metrics()
    
    batch_stats_callback = CollectBatchStats()

# train the model
    top_model.fit(
        train_data, train_labels,
        nb_epoch = 50, 
        batch_size = 128,
        validation_data=(validation_data, validation_labels),
        verbose = 1,
        callbacks = [batch_stats_callback]
    )

#export the model
    top_model.save("/media/sf_VirtualBox_Share/top_model.h5")

#
    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0,2])
    plt.plot(batch_stats_callback.batch_losses)
    plt.savefig("/media/sf_VirtualBox_Share/Training_loss.png")


def combine_model():
#load the top_model
    top_model = keras.models.load_model("/media/sf_VirtualBox_Share/top_model.h5")

#define the full_model
    final_model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
                   trainable=False),  
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    final_model.build([None, 299, 299, 3])

# Combine models - assign the weights from top_model
    for i in [-3, -2, -1]:
        weights = top_model.layers[i].get_weights()
        final_model.layers[i].set_weights(weights)

    return final_model


def show_prediction(input_data, predicted_batch):
#
    class_names = sorted(input_data.class_indices.items(), key=lambda pair:pair[1])
    class_names = np.array([key.title() for key, value in class_names])
#
    for image_batch, label_batch in input_data:
#        print("Image batch shape: ", image_batch.shape)
#        print("Label batch shape: ", label_batch.shape)
        break 
#
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]
    label_id = np.argmax(label_batch, axis=-1)
    plt.figure(figsize=(10,9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6,5,n+1)
        plt.imshow(image_batch[n])
        color = "green" if predicted_id[n] == label_id[n] else "red"
        plt.title(predicted_label_batch[n].title(), color=color)
        plt.axis('off')
    _   = plt.suptitle("Model predictions (green: correct, red: incorrect)")
    plt.show()


def call_lime(predit_data, i=0):
#    
    class_names = sorted(predit_data.class_indices.items(), key=lambda pair:pair[1])
    class_names = np.array([key.title() for key, value in class_names])
#
    final_model = combine_model()

# print prediction result
    predicted_batch = final_model.predict(predit_data)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    label_id = np.argmax(label_batch, axis=-1)
    print("Actual Style is: ", class_names[label_id[i]])
    for j in range (0, 7, 1):
        print("Prediction for ", class_names[j], ": ", predicted_batch[i,j]) 
#
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_batch[i], final_model.predict, 
        top_labels=5, hide_color=0, num_samples=1000
        )
# show the top class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    color = "green" if predicted_id[i] == label_id[i] else "red"
    plt.title(class_names[label_id[i]], color=color)
    plt.suptitle("Model predictions (green: correct, red: incorrect)")
#    plt.show()
#
    return explanation


# ______main_______

#save_bottleneck()
#train_top_model()

# load prediction data
predict_data = datagen.flow_from_directory(
        predict_dir,
        target_size=IMAGE_SHAPE,
        class_mode='categorical',
        shuffle=False
        )
predict_data.batch_size = predict_data.classes.size

# check predition data
for image_batch, label_batch in predict_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break 

# show prediction result
final_model = combine_model()
final_model.summary()
# predicted_batch = final_model(image_batch)
# show_prediction(predict_data, predicted_batch)

# Lime - open the black box - can only import 1 image each time
for i in range(0, predict_data.batch_size+1, 1):
    explanation = call_lime(predict_data, i)
    plt.savefig("/media/sf_VirtualBox_Share/Lime_outcome/" + str(i) + ".png")





