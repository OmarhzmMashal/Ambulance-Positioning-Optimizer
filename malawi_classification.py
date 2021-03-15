import os
folder = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import pandas as pd 
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from sklearn.model_selection import KFold
from collections import Counter

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
import re
from tensorflow import keras

from category_encoders import TargetEncoder


# cut long texts and create now rows (not used)
def cut(int2texts):  
    extended_texts=[]
    extended_labels=[]  
    for idx, text in int2texts.items():
        short_text=""
        if (len(text.split())) > 200:
            for word_idx in range(len(text.split())):
                short_text += " " + text[word_idx]
                if word_idx == int(len(text.split())/2):
                    extended_texts.append(short_text)
                    extended_labels.append(int2labels[idx])
                    short_text = ""
                if word_idx == len(text.split())-1:
                    extended_texts.append(short_text)
                    extended_labels.append(int2labels[idx])
                    short_text = ""            
        else:
            extended_texts.append(text)
            extended_labels.append(int2labels[idx])
            
    return extended_texts, extended_labels

def clean_txt(text):
    text = text.lower()
    text = re.sub(r"[-()*&^%$#@!,0123456789<>|.?+=]", "", text)
    return text

# read data
training = pd.read_csv(folder + "/Train.csv")
testing = pd.read_csv(folder+"/Test.csv")

texts = training['Text']
labels = training['Label']

print(labels.value_counts().plot(kind='bar'))
plt.show()
#print(labels.unique())


# create dict containing cat to text
text2cat={}
i=0
for t in texts:
    text2cat[t] = labels[i]
    i+=1


cat2list_words ={}
word_list=[]
count=0
# store words of each cat in every catg
for text, cat in text2cat.items():
    for w in clean_txt(text).split():        
        if cat in cat2list_words:
            cat2list_words[cat].append(w)  
        else:
            cat2list_words[cat] = []
    

# create a set of words for each cat
#for text, list_words in text2cat.items():       
#    cat2list_words[cat] = set(cat2list_words[cat])
    
    
# dict for over sampling assigning the number of samples for each cat 
cat2counts = labels.value_counts().to_dict()  
cat2prec1 ={}
for cat, count in cat2counts.items():
    if cat in ['POLITICS', 'SOCIAL', 'RELIGION', 'LAW/ORDER', 'SOCIAL ISSUES']:
        cat2prec1[cat] = 279


# clean text 
int2texts={}
int2labels={}
for t in range(len(texts)):
    int2texts[t] = clean_txt(texts[t])
    int2labels[t] = labels[t]
    
# count num of words
word2count={}
for idx, text in int2texts.items():
    for w in text.split():
        if w not in word2count:
            word2count[w] = 1
        else:
            word2count[w] += 1
            

# eleminate non freq words
threshold =3
elim_texts =[]
short_text=""
new_word2count={}
for idx, text in int2texts.items():
    for word in text.split():        
        if word2count[word] > threshold:
            short_text += " " + word
            new_word2count[word] = word2count[word]
    elim_texts.append(short_text)
    short_text=""



num_words=len(new_word2count) + 1 #the oov token
tokenize = keras.preprocessing.text.Tokenizer(num_words=int(num_words), 
                                              char_level=False,oov_token="<OOV>")

# fit tokenizer to our training text data
tokenize.fit_on_texts(texts) 

#convert to matrix accurence count and binary as features
x_train1 = tokenize.texts_to_matrix(texts,mode='binary')
x_train2 = tokenize.texts_to_matrix(texts,mode='count')    
x_train =  np.concatenate((x_train1, x_train2), axis=1)

x_test1 = tokenize.texts_to_matrix(testing['Text'],mode='binary')
x_test2 = tokenize.texts_to_matrix(testing['Text'],mode='count')
x_test = np.concatenate((x_test1, x_test2), axis=1)


# create num of words as a feature
test_word_counts=[]
for i in range(len(x_test)):
    test_word_counts.append(np.sum(x_test[i]))

train_word_counts=[]
for i in range(len(x_train)):
    train_word_counts.append(np.sum(x_train[i]))

test_word_counts = np.asarray(test_word_counts)
train_word_counts = np.asarray(train_word_counts)

test_word_counts=test_word_counts.reshape(len(test_word_counts),1)
train_word_counts=train_word_counts.reshape(len(train_word_counts),1)

# add num of words as a feature
x_test = np.concatenate((x_test, test_word_counts), axis=1)
x_train = np.concatenate((x_train, train_word_counts), axis=1)
    


# resample the data
os = RandomOverSampler(sampling_strategy=cat2prec1) 
#us = RandomUnderSampler(sampling_strategy=cat2prec1)


#x_train, labels = us.fit_resample(x_train, labels)
x_train, labels = os.fit_resample(x_train, labels)


# plot new dist
print(labels.value_counts().plot(kind='bar'))
plt.show()


# ordinal encoding
encoder = LabelEncoder()
encoder.fit(labels)
y_train = encoder.transform(labels)

# then on hot encoding
num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)


# convert to sequence of integers represention the sentance
x_train_seq = tokenize.texts_to_sequences(texts)
x_test_seq = tokenize.texts_to_sequences(testing['Text'])
x_train_seq = pad_sequences(x_train_seq)
x_test_seq = pad_sequences(x_test_seq)


#hyper paramaters
batch_size = 32
epochs = 30
drop_ratio = 0.2
embedding_dim = 16
max_length = len(x_train_seq[0])


'''
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
'''


layers = keras.layers
models = keras.models

model = models.Sequential()
model.add(layers.Dense(512))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(drop_ratio))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))


'''
model = models.Sequential()
model.add(tf.keras.layers.Dense(128,input_shape=(x_train[0].shape)))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(layers.Dropout(drop_ratio))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
'''

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

def train_seq():
    history = model.fit(x_train_seq, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

def train_matrix():
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
    

def submit():
    # predict each text in test
    text_labels = encoder.classes_ 
    arr = []
    
    for i in range(len(x_test)):
        prediction = model.predict(np.array([x_test[i]]))
        predicted_label = text_labels[np.argmax(prediction)]
        arr.append(predicted_label)
    
    # set predictions to label column
    testing = pd.read_csv(folder+"/SampleSubmission.csv")
    testing['Label'] = arr
    
    # save csv file
    testing.to_csv(folder+"/Sub.csv",index=False)


if __name__ == '__main__':
    train_matrix()
    submit()