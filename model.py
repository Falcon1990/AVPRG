import tensorflow as tf 
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import matplotlib.pyplot as plt

enc = LabelEncoder()
ssc = StandardScaler()

#transformiert die csv daten zu arrays und Skaliert sie entlang der Axe 0
def preprocessing(data_set):

  # kodiert unsere labels also klassen von zu string zu int
  genres = data_set.iloc[:, -1]
  labels = enc.fit_transform(genres)

  # Skaliert die Werte in der Feature-Spalte so das sie eine ähnliche Standartverteilung besitzen
  features = ssc.fit_transform(np.array(data_set.iloc[:, :-1], dtype = float))

  #Aufteilen des Data-Set in sub-sets
  train_features = features[:int(0.6*len(features))]
  train_labels = labels[:int(0.6*len(labels))]

  val_features = features[int(0.6*len(features)):int(0.8*len(features))]
  val_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]

  test_features = features[int(0.8*len(features)):]
  test_labels = labels[int(0.8*len(labels)):]

  print('Scaling und Spliting fertig')

  return train_features, train_labels, test_features, test_labels, val_features, val_labels 


def model(train_features, train_labels, test_features, test_labels, val_features, val_labels):

  model = Sequential()
  model.add(Dense(512, activation = 'relu', input_shape=(train_features.shape[1],)))

  model.add(Dropout(0.5))

  model.add(Dense(256, activation='relu', input_shape=(train_features.shape[1],)))

  model.add(Dropout(0.5))

  model.add(Dense(128, activation='relu'))

  model.add(Dense(64, activation='relu'))

  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  history = model.fit(train_features,
            train_labels,
            epochs=150,
            batch_size= 10, 
            validation_data=(val_features, val_labels))

  #Accuracy Graph
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

  #Loss Graph
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()
    
  evaluate = model.evaluate(test_features, test_labels)

  model.save_weights('AVPRG_MODEL_WEIGHTS.h5')
  model.save('AVPRG_MODEL.h5')

  return evaluate

def arg_err():
    print ('Benutzung-> python preproccess.py [Pfad zu .csv Dateien]')
    sys.exit(0)

if __name__ == "__main__":

  if len(sys.argv) < 2:
        arg_err()

  # uncomment um alle csv dateien aus dem in der bash eingegebenen Directory zu laden (lädt nur die Pfade) 
  #data_sets_paths = glob.glob(data_set_path + '/*.csv')

  data_set_path = sys.argv[1]
  file = open(data_set_path, 'r', newline='')
  with file:

    data_set = pd.read_csv(file)

    train_features, train_labels, test_features, test_labels, val_features, val_labels = preprocessing(data_set)

    result = model(train_features, train_labels, test_features, test_labels, val_features, val_labels)

    print (result)


  

######################################################################
# TFRecord Proto-code
#
# nicht funktionsfähig

# feature_vector = {  'label': tf.FixedLenFeature([], tf.int64),
#                     'chroma_sftf': tf.VarLenFeature(tf.float32),
#                     'mfcc_0': tf.VarLenFeature(tf.float32),
#                     'mfcc_1': tf.VarLenFeature(tf.float32),
#                     'mfcc_2': tf.VarLenFeature(tf.float32),
#                     'mfcc_3': tf.VarLenFeature(tf.float32),
#                     'mfcc_4': tf.VarLenFeature(tf.float32),
#                     'mfcc_5': tf.VarLenFeature(tf.float32),
#                     'mfcc_6': tf.VarLenFeature(tf.float32),
#                     'mfcc_7': tf.VarLenFeature(tf.float32),
#                     'mfcc_8': tf.VarLenFeature(tf.float32),
#                     'mfcc_9': tf.VarLenFeature(tf.float32),
#                     'mfcc_10': tf.VarLenFeature(tf.float32),
#                     'mfcc_11': tf.VarLenFeature(tf.float32),
#                     'mfcc_12': tf.VarLenFeature(tf.float32),
#                     'mfcc_13': tf.VarLenFeature(tf.float32),
#                     'mfcc_14': tf.VarLenFeature(tf.float32),
#                     'mfcc_15': tf.VarLenFeature(tf.float32),
#                     'mfcc_16': tf.VarLenFeature(tf.float32),
#                     'mfcc_17': tf.VarLenFeature(tf.float32),
#                     'mfcc_18': tf.VarLenFeature(tf.float32),
#                     'mfcc_19': tf.VarLenFeature(tf.float32),
#                     'spect_centroid': tf.VarLenFeature(tf.float32),
#                     'spect_flatness': tf.VarLenFeature(tf.float32),
#                     'spect_bandwidth': tf.VarLenFeature(tf.float32),
#                     'zero_crossing_rate': tf.VarLenFeature(tf.float32),
#                     'spect_rolloff': tf.VarLenFeature(tf.float32)
#            }

# def create_sapmle(example_sample, clip=False):

#     example = tf.parse_single_example(example_sample, feature_vector)
#     label = tf.cast(example['label'], tf.int32)
#     chmstft= example['chroma_sftf']
#     mfcc_1 = example['mfcc_0']
#     mfcc_2 = example['mfcc_1']
#     mfcc_3 = example['mfcc_2']
#     mfcc_4 = example['mfcc_3']
#     mfcc_5 = example['mfcc_4']
#     mfcc_6 = example['mfcc_5']
#     mfcc_7 = example['mfcc_6']
#     mfcc_8 = example['mfcc_7']
#     mfcc_9 = example['mfcc_8']
#     mfcc_10 = example['mfcc_9']
#     mfcc_11 = example['mfcc_10']
#     mfcc_12 = example['mfcc_11']
#     mfcc_13 = example['mfcc_12']
#     mfcc_14 = example['mfcc_13']
#     mfcc_15 = example['mfcc_14']
#     mfcc_16 = example['mfcc_15']
#     mfcc_17 = example['mfcc_16']
#     mfcc_18 = example['mfcc_17']
#     mfcc_19 = example['mfcc_18']
#     mfcc_20 = example['mfcc_19']
#     spctcent= example['spect_centroid']
#     spctfltns = example['spect_flatness']
#     spctbdnwt = example['spect_bandwidth']
#     zrocrssrt = example['zero_crossing_rate']
#     spctrllf= example['spect_rolloff']

#     return label, chmstft, mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7, mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, mfcc_13, mfcc_14, mfcc_15, mfcc_16, mfcc_17, mfcc_18, mfcc_19, mfcc_20, spctcent, spctbdnwt, spctfltns, zrocrssrt, spctrllf

# feature_set = tf.data.TFRecordDataset('AVPRG_train.tfrecord').map(create_sapmle)
# feature_set = feature_set.shuffle(200).batch(32)

# feature_label = feature_set['label']
