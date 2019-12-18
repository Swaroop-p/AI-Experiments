# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:34:29 2019

@author: Swaroop.Padala
"""
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def prepare_input_for_model(n, pad_seq_len):
    return pad_sequences(np.array([list(map(int, list(convertToBinary(n))))]), maxlen=pad_seq_len)
    
def predict_survivor(n, model):
    test_input = prepare_input_for_model(n, model.layers[0].input_shape[1])
    prediction = list(int(round(x)) for x in model.predict(test_input)[0])    
    test_output_in_binary = int("".join(map(str,prediction)))    
    result = int(str(test_output_in_binary), 2)
    print("Input: ", n, "\n", "output: " , result)
    
def msbPos(n): 
    pos = 0
    while n != 0: 
        pos += 1
        n = n >> 1
    return pos 
  
def convertToBinary(n):
   return bin(n).replace("0b","") 

def josephify(n): 
    position = msbPos(n)  
    j = 1 << (position - 1) 
    n = n ^ j 
    n = n << 1
    n = n | 1
    return n 

def train_josephus_DL_model(n=1000):
    inputs = []
    outputs = []
    for i in range(1,n):
        #inputs.append(i)
        inputs.append(convertToBinary(i))
        outputs.append(convertToBinary(josephify(i)))
    
    inp = [list(n) for n in inputs]
    out = [list(n) for n in outputs]
    
    inp_int = [list(map(int,x)) for x in inp]
    out_int = [list(map(int,x)) for x in out]
    
    X = pad_sequences(inp_int)
    #X = inputs
    y = pad_sequences(out_int)
    
    input_nodes_len = len(X[0])
    #input_nodes_len = 1
    output_nodes_len = len(y[0])
    
    model = Sequential()
    model.add(Dense(24, input_dim=input_nodes_len, activation='relu'))
    model.add(Dense(24,  activation='relu'))
    model.add(Dense(24,  activation='relu'))
    model.add(Dense(output_nodes_len,  activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model.fit(X, y, epochs=60, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))
    
    return model

tf_model = train_josephus_DL_model(3000)

predict_survivor(2500, tf_model)