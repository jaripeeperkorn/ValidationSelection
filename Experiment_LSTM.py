import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Bidirectional, LSTM, Dropout, Input
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
import pandas as pd

def save_log(loglist, filename): #save a list of lists 
    df = pd.DataFrame.from_records(loglist)
    df.to_csv(filename, index=False)

def get_model(maxlen, num_chars, bidirec, n_layers, lstmsize, dropout, l1, l2):
    model = Sequential()
    model.add(Input(shape=(maxlen, num_chars))) #If you don't use an embedding layer input should be one-hot-encoded
    if bidirec == False:   
        model.add(LSTM(lstmsize,kernel_initializer='glorot_uniform',return_sequences=(n_layers != 1),kernel_regularizer=regularizers.l1_l2(l1,l2),
                       recurrent_regularizer=regularizers.l1_l2(l1,l2),input_shape=(maxlen, num_chars)))
        model.add(Dropout(dropout))
        for i in range(1, n_layers):
            return_sequences = (i+1 != n_layers)
            model.add(LSTM(lstmsize,kernel_initializer='glorot_uniform',return_sequences=return_sequences,kernel_regularizer=regularizers.l1_l2(l1,l2),recurrent_regularizer=regularizers.l1_l2(l1,l2)))
            model.add(Dropout(dropout))
    else:
        model.add(Bidirectional(LSTM(lstmsize,kernel_initializer='glorot_uniform',return_sequences=(n_layers != 1),kernel_regularizer=regularizers.l1_l2(l1,l2),recurrent_regularizer=regularizers.l1_l2(l1,l2),input_shape=(maxlen, num_chars))))
        model.add(Dropout(dropout))
        for i in range(1, n_layers):
            return_sequences = (i+1 != n_layers)
            model.add(Bidirectional(LSTM(lstmsize,kernel_initializer='glorot_uniform',return_sequences=return_sequences,kernel_regularizer=regularizers.l1_l2(l1,l2),recurrent_regularizer=regularizers.l1_l2(l1,l2))))
            model.add(Dropout(dropout))
    model.add(Dense(num_chars, kernel_initializer='glorot_uniform',activation='softmax'))
    opt = Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics='accuracy')
    return model

def train_model(X_train, y_train,batch_size, maxlen, num_chars, bidirec, n_layers, lstmsize, dropout, l1, l2, x_val, y_val):
    model = get_model(maxlen, num_chars, bidirec, n_layers, lstmsize, dropout, l1, l2)
    #model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    #train_model
    history = model.fit(X_train, y_train, callbacks=[early_stopping, lr_reducer], batch_size=batch_size, 
                        epochs=600, verbose=1, validation_data=(x_val, y_val))
    return model


def number_to_one_hot_X(X, dict_size): #if we want 
    newX = []
    for example in X:
        new_ex = []
        for i in range(len(example)):
            onehot = [0]*dict_size #changed
            if example[i] != 0:
                onehot[example[i] - 1] = 1 #-1 because begin counting at 0
            new_ex.append(onehot)
        newX.append(new_ex)
    return(np.array(newX))

def create_XY_prefix(log, mappingsize, prefixlen):
    X = []
    Y = []
    for i in range(0, len(log)):
        for k in range(1, len(log[i])):
            X.append(log[i][max(0, k-prefixlen):k]) #get the prefix of 'encoded' activities
            y = [0] *(mappingsize)
            y[int(log[i][k])-1] = 1
            Y.append(y)        
    X = keras.utils.pad_sequences(X, maxlen=prefixlen, padding='pre')
    X = number_to_one_hot_X(X, mappingsize)
    return(np.array(X), np.array(Y))


def get_startend(log): 
    return log[0][0], log[0][-1]
def cut_end(log, endact):
    logsize, tracesize = log.shape
    newlog = []
    for i in range(0, logsize):
        trace = []
        for j in range(0, tracesize):
            if log[i][j] == endact:
                trace.append(log[i][j])
                break
            else:
                trace.append(log[i][j])
        newlog.append(trace)
    return(newlog)


def normalize(probs): #normalize probabilities to sum to 1
    examplesize, actsize = probs.shape
    newy = []
    for i in range(examplesize):
        normalizer = 1 / float( sum(probs[i]) )
        ynorm = [float(l) * normalizer for l in probs[i]]
        newy.append(ynorm)
    return newy


def choose_act_all(all_y): #randomly choose an activity, stochastically
  #p want a list of probabilities    
    chosen_acts = []
    for i in range(len(all_y)):
        chosen_acts.append(np.random.choice(np.arange(0, len(all_y[i])), p=all_y[i])+1)  
    return(chosen_acts)   # +1 because number encodig starts at 1 not 0

def get_probabilities(rnnmodel, xlists,  nr_act, prefixlen):
    #assume xlist is a list with the x (prefix) untill now 
    all_x = keras.utils.pad_sequences(xlists, maxlen=prefixlen, padding="pre")
    all_x = all_x[:,-(prefixlen):]
    all_x = number_to_one_hot_X(all_x, nr_act)
    results = rnnmodel.predict(all_x)
    return results

def simulate_log(RNNmodel, logsize, startact, endact, maxlen, vocsize, prefixlen): #Use RNN to simulate log
    log = np.zeros((logsize, maxlen), int)
    for i in range(0, logsize): #start every trace with the start activity
        log[i][0] = startact
    for j in range(1,maxlen): #check if 0 or 1 and ml or ml - 1 #we took 50 for with loops   
        print("finding activity nr", j+1)   
        prefixes = np.array([log[i][0:j] for i in range(0, logsize)])
        probs = get_probabilities(RNNmodel, prefixes, vocsize,  prefixlen)
        #we need to do this because otherwise probabilities sum over 1 
        ynorm = normalize(probs) 
        nextacts = choose_act_all(ynorm) 
        for i in range(0, logsize):
            log[i][j] = nextacts[i]
    corrected_log = cut_end(log, endact)      
    return(corrected_log) 


def evaluate_and_simulate(model, X_test, y_test,start,end,size_simulated_log, maximumlength, mapping, prefixlen,
                          location_simulated_log): 
    #we assume the cases have a fixed start and end activity
    loss_t, accuracy_t = model.evaluate(X_test, y_test)
    print('Test Loss: ', loss_t)
    print('Test accuracy: ', accuracy_t)
    simlog = simulate_log(model, size_simulated_log, start, end, maximumlength, len(mapping), prefixlen)
    save_log(simlog, location_simulated_log+'.csv')

#mapping saves map from activity to number    
def do_experiment(train_log, val_log, test_log, mapping,
                  size_simulated_log, location_simulated_log,
                  prefixlen, maximumlength, bidirec, n_layers, lstmsize, dropout, l1, l2,
                  batch_size):
    
    if prefixlen == False:
         prefixlen = len(max(train_log,key=len))
         
    X_train, y_train = create_XY_prefix(train_log, len(mapping), prefixlen)
    X_val, y_val = create_XY_prefix(val_log, len(mapping), prefixlen)
    X_test, y_test = create_XY_prefix(test_log, len(mapping), prefixlen)
    
    model = train_model(X_train, y_train,batch_size, maxlen=prefixlen, num_chars=len(mapping), 
                        bidirec=bidirec, n_layers=n_layers, lstmsize=lstmsize, dropout=dropout, l1=l1, l2=l2, 
                        x_val=X_val, y_val=y_val)
    start,end = get_startend(train_log)
    evaluate_and_simulate(model, X_test, y_test,start,end,size_simulated_log, maximumlength, mapping, prefixlen,
                              location_simulated_log)