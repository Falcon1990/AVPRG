from random import shuffle
import glob
import librosa as li
import numpy as np
import tensorflow as tf
import sys
import csv
import os

shuffle_data = True

def _extract_Music_Features(filename):

    #mfcc = []
    feature_set = []

    y, sr = li.load(filename, mono=True, duration=30)
    #generiert Spectogram (mel scale) 
    S = li.feature.melspectrogram(y=y, sr=sr)
    # konvertiert energie zu dB
    log_S = li.power_to_db(S, ref=np.max)
    
    #Extrahieren von verschiedenen Musik features nach Librosa(audio manipulation framework)
    chr_sftf = li.feature.chroma_stft(y=y, sr=sr)
    chr_sftf = np.mean(chr_sftf)
    feature_set.append(chr_sftf)

    _mfcc = li.feature.mfcc(S=log_S, sr=sr)

    spect_cent = li.feature.spectral_centroid(y=y, sr=sr)
    spect_cent = np.mean(spect_cent)
    feature_set.append(spect_cent)

    spect_bndwth = li.feature.spectral_bandwidth(y=y, sr=sr)
    spect_bndwth= np.mean(spect_bndwth)
    feature_set.append(spect_bndwth)

    zro_crs_rt = li.feature.zero_crossing_rate(y=y)
    zro_crs_rt = np.mean(zro_crs_rt)
    feature_set.append(zro_crs_rt)

    spect_rllf = li.feature.spectral_rolloff(y=y, sr=sr)
    spect_rllf = np.mean(spect_rllf)
    feature_set.append(spect_rllf)

    spect_fltns = li.feature.spectral_flatness(y=y)
    spect_fltns = np.mean(spect_fltns)
    feature_set.append(spect_fltns)

    for i in _mfcc:
        x = np.mean(i)
        #mfcc.append(x)
        feature_set.append(x)
    
    label = os.path.basename(filename)
    label = label.split('-')[0]
    feature_set.append(label)

    #chr_sftf, mfcc, spect_cent, spect_bndwth, zro_crs_rt, spect_rllf, spect_fltns

    return feature_set 


def arg_err():
    print ('Benutzung-> python preproccess.py [Pfad zu wav Dateien] [Pfad zum speichern der TFRecord File]')
    sys.exit(0)

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        arg_err()

    #bash args für directories 
    wav_path = sys.argv[1]
    #tfrecord = 'AVPRG.tfrecords'

    #lade alle wavs pfade im ordner
    wavs = glob.glob(wav_path + '/*.wav')
    labels = [0 if 'Klassik' in wav else 1 for wav in wavs]

    #csv zeugs
    csv_headers = 'chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate spect_flatness'
    for i in range(1, 21):
        csv_headers += f' mfcc{i}'
    csv_headers += ' label'
    csv_headers = csv_headers.split()

    file = open('avprg_data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)

    # mischen gegen möglichen bias
    if shuffle_data:
        e = list(zip(wavs, labels))
        shuffle(e)
        wavs, labels = zip(*e)

    for i in range(len(wavs)):

        print('adding to csv: ' + wavs[i] )

        file = open('avprg_data.csv', 'a', newline='')
        with file:

            features = _extract_Music_Features(wavs[i])

            writer = csv.writer(file)
            writer.writerow(features)



    ######################################################################
    # Speichern und verwalten des Datensatzes mittels Tensorflow TFRecords 
    #
    # unausgereift und nicht ganz funktionsfähig -> wie kann den Datensatz in Keras einfügen? 

    # öffne TFRecord Datei
    # writer = tf.python_io.TFRecordWriter(tfrecord)

    # #iteriere über die wavs pfade in der liste
    # for i in range(len(train_wavs)):

    #     print ('preprocessing train data' + train_wavs[i])
    #     #Musik Feature extraction
    #     chr_sftf, mfcc, spect_cent, spect_bndwth, zro_crs_rt, spect_rllf, spect_fltns = _extract_Music_Features(train_wavs[i])

    #     #Label anfügen
    #     label = train_labels[i]

    #     # feature vektor erzeugen für TFRecord
    #     feature = { 'label': int64_feature(label),
    #                 'chroma_sftf': float_feature(chr_sftf),
    #                 'mfcc_0': float_feature(mfcc[0]),
    #                 'mfcc_1': float_feature(mfcc[1]),
    #                 'mfcc_2': float_feature(mfcc[2]),
    #                 'mfcc_3': float_feature(mfcc[3]),
    #                 'mfcc_4': float_feature(mfcc[4]),
    #                 'mfcc_5': float_feature(mfcc[5]),
    #                 'mfcc_6': float_feature(mfcc[6]),
    #                 'mfcc_7': float_feature(mfcc[7]),
    #                 'mfcc_8': float_feature(mfcc[8]),
    #                 'mfcc_9': float_feature(mfcc[9]),
    #                 'mfcc_10': float_feature(mfcc[10]),
    #                 'mfcc_11': float_feature(mfcc[11]),
    #                 'mfcc_12': float_feature(mfcc[12]),
    #                 'mfcc_13': float_feature(mfcc[13]),
    #                 'mfcc_14': float_feature(mfcc[14]),
    #                 'mfcc_15': float_feature(mfcc[15]),
    #                 'mfcc_16': float_feature(mfcc[16]),
    #                 'mfcc_17': float_feature(mfcc[17]),
    #                 'mfcc_18': float_feature(mfcc[18]),
    #                 'mfcc_19': float_feature(mfcc[19]),
    #                 'spect_centroid': float_feature(spect_cent),
    #                 'spect_flatness': float_feature(spect_fltns),
    #                 'spect_bandwidth': float_feature(spect_bndwth),
    #                 'zero_crossing_rate': float_feature(zro_crs_rt),
    #                 'spect_rolloff': float_feature(spect_rllf)
    #                }

    #     # Schreibt den Serialized Feature Vector in die TFRecord File
    #     example = tf.train.Example(features=tf.train.Features(feature=feature))
        
    #     writer.write(example.SerializeToString())
        
    # writer.close()
    # sys.stdout.flush()

    # #öffne TFRecord Datei
    # writer_2 = tf.python_io.TFRecordWriter(tfrecord_test)

    # for i in range(len(test_wavs)):

    #     print('preprocessing test data ' + test_wavs[i])
    #     #Musik Feature extraction
    #     chr_sftf, mfcc, spect_cent, spect_bndwth, zro_crs_rt, spect_rllf, spect_fltns = _extract_Music_Features(test_wavs[i])

    #     #Label anfügen
    #     label = test_labels[i]

    #     # feature vektor erzeugen für TFRecord
    #     feature = { 'label': int64_feature(label),
    #                 'chroma_sftf': float_feature(chr_sftf),
    #                 'mfcc_0': float_feature(mfcc[0]),
    #                 'mfcc_1': float_feature(mfcc[1]),
    #                 'mfcc_2': float_feature(mfcc[2]),
    #                 'mfcc_3': float_feature(mfcc[3]),
    #                 'mfcc_4': float_feature(mfcc[4]),
    #                 'mfcc_5': float_feature(mfcc[5]),
    #                 'mfcc_6': float_feature(mfcc[6]),
    #                 'mfcc_7': float_feature(mfcc[7]),
    #                 'mfcc_8': float_feature(mfcc[8]),
    #                 'mfcc_9': float_feature(mfcc[9]),
    #                 'mfcc_10': float_feature(mfcc[10]),
    #                 'mfcc_11': float_feature(mfcc[11]),
    #                 'mfcc_12': float_feature(mfcc[12]),
    #                 'mfcc_13': float_feature(mfcc[13]),
    #                 'mfcc_14': float_feature(mfcc[14]),
    #                 'mfcc_15': float_feature(mfcc[15]),
    #                 'mfcc_16': float_feature(mfcc[16]),
    #                 'mfcc_17': float_feature(mfcc[17]),
    #                 'mfcc_18': float_feature(mfcc[18]),
    #                 'mfcc_19': float_feature(mfcc[19]),
    #                 'spect_centroid': float_feature(spect_cent),
    #                 'spect_flatness': float_feature(spect_fltns),
    #                 'spect_bandwidth': float_feature(spect_bndwth),
    #                 'zero_crossing_rate': float_feature(zro_crs_rt),
    #                 'spect_rolloff': float_feature(spect_rllf)
    #                }

    #     # Schreibt den Serialized Feature Vector in die TFRecord File
    #     example = tf.train.Example(features=tf.train.Features(feature=feature))
        
    #     writer_2.write(example.SerializeToString())
        
    # writer_2.close()
    # sys.stdout.flush()

    
    # #öffne TFRecord Datei
    # writer_3 = tf.python_io.TFRecordWriter(tfrecord_val)

    # for i in range(len(val_wavs)):

    #     print('preprocessing test data ' + val_wavs[i])
    #     #Musik Feature extraction
    #     chr_sftf, mfcc, spect_cent, spect_bndwth, zro_crs_rt, spect_rllf, spect_fltns = _extract_Music_Features(val_wavs[i])

    #     #Label anfügen
    #     label = val_labels[i]

    #     # feature vektor erzeugen für TFRecord
    #     feature = { 'label': int64_feature(label),
    #                 'chroma_sftf': float_feature(chr_sftf),
    #                 'mfcc_0': float_feature(mfcc[0]),
    #                 'mfcc_1': float_feature(mfcc[1]),
    #                 'mfcc_2': float_feature(mfcc[2]),
    #                 'mfcc_3': float_feature(mfcc[3]),
    #                 'mfcc_4': float_feature(mfcc[4]),
    #                 'mfcc_5': float_feature(mfcc[5]),
    #                 'mfcc_6': float_feature(mfcc[6]),
    #                 'mfcc_7': float_feature(mfcc[7]),
    #                 'mfcc_8': float_feature(mfcc[8]),
    #                 'mfcc_9': float_feature(mfcc[9]),
    #                 'mfcc_10': float_feature(mfcc[10]),
    #                 'mfcc_11': float_feature(mfcc[11]),
    #                 'mfcc_12': float_feature(mfcc[12]),
    #                 'mfcc_13': float_feature(mfcc[13]),
    #                 'mfcc_14': float_feature(mfcc[14]),
    #                 'mfcc_15': float_feature(mfcc[15]),
    #                 'mfcc_16': float_feature(mfcc[16]),
    #                 'mfcc_17': float_feature(mfcc[17]),
    #                 'mfcc_18': float_feature(mfcc[18]),
    #                 'mfcc_19': float_feature(mfcc[19]),
    #                 'spect_centroid': float_feature(spect_cent),
    #                 'spect_flatness': float_feature(spect_fltns),
    #                 'spect_bandwidth': float_feature(spect_bndwth),
    #                 'zero_crossing_rate': float_feature(zro_crs_rt),
    #                 'spect_rolloff': float_feature(spect_rllf)
    #                }

    #     # Schreibt den Serialized Feature Vector in die TFRecord File
    #     example = tf.train.Example(features=tf.train.Features(feature=feature))
        
    #     writer_3.write(example.SerializeToString())
        
    # writer_3.close()
    # sys.stdout.flush()

    
