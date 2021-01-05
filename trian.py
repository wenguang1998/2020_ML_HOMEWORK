#!/usr/bin/env python
# coding: utf-8

# In[5]:



# import required packages
import tensorflow as tf
import tensorflow_datasets as tfds  # library to import the IMDB dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import code

if __name__ == "__main__":
    # 1. load your training data
    # obtaining the IMDB reviews dataset for semantic analysis
    # downloading the dataset
#    train_data, test_data = tfds.load(name='imdb_reviews', split=('train', 'test'), as_supervised=True)
    train_data = pd.read_csv(r"C:\Users\guang\Desktop\study5be\ml\personal\2\public_data\train.csv",encoding=u'utf-8')
    train_data.info()
    print(train_data.describe())
    # creating empty list for training set
    train_sent = []
    train_out = []
    
    
    for i in range(0,40000):
        train_sent.append(train_data['review'].values[i])
        if train_data['sentiment'].values[i] == 'positive':
            train_out.append(1)
        else:
            train_out.append(0)
    # for loop to obtain train set from the downloaded dataset
#     for s, l in train_data:
#         print(str(s.numpy()))
#         print(l.numpy())
#         train_sent.append(str(s.numpy()))
#         train_out.append(l.numpy())

    # converting the label dataset to array formate
    train_out = np.array(train_out)
#     for i in range(0,5):
#         print(train_sent[i])
#         print(train_out[i])

    test_data = pd.read_csv(r"C:\Users\guang\Desktop\study5be\ml\personal\2\public_data\test_data.csv",encoding = u'utf-8')
    
    test_sent = []
    for i in range(0,10000):
        test_sent.append(test_data['review'].values[i])

    # hyper parameters - writing them at one place so that it's easy to modify them in future
    vocab_size = 20000  # 
    embedding_dim = 128   
    max_length = 400     
    trunc_type = 'post'
    oov_tok = "<OOV>"
    num_epochs = 2

    # initializing the tokenizer using hyper parameters.
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok) # number of words for the number of words to tokenize, oov_token is for unknown words 
    tokenizer.fit_on_texts(train_sent)  # applying the tokenizer on the input texts
    word_index = tokenizer.word_index   # obtaining the word index
    sequences = tokenizer.texts_to_sequences(train_sent)    # converting the input texts to their relative word_indexes
    padded = pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)   # padding to make all the input texts of the same length
    
    test_sent = tokenizer.texts_to_sequences(test_sent) #tokenizing the test dataset
    test_padded = pad_sequences(test_sent,maxlen=max_length,truncating=trunc_type)    # padding the test dataset
    # 2. Train your network
    # GRU and Conv1D
    # Model Definition with Conv1D
    model = tf.keras.Sequential([
        # using embedding layer as the first layer
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        # Adding Conv1D layer as the second layer
        tf.keras.layers.Conv1D(128, 5, activation='relu'),  

        # adding bidirectional GRU layer
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)), 
            
        # Adding small fully connected layer
        tf.keras.layers.Dense(6, activation='relu'),   

        # adding the final output node with sigmoid activation function
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # compiling the model using binary_crossentropy loss & Adam optimizer & accuracy as the metric
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    # printing the model summary to see what is the final model that we have 
    model.summary()

    # Training the model, using validation data to obtain teh accuracy after each epoches
    history = model.fit(padded, train_out, epochs=num_epochs)
    
    pred = model.predict(test_padded)
    print(len(pred))
    print(pred[0])
    re = open(r"C:\Users\guang\Desktop\re.txt",'w')
    for i in range(0,10000):
        if pred[i] > 0.5:
            re.write("positive\n")
        else:
            re.write("negtive\n")
    # 3. Save your model
    


# In[ ]:




