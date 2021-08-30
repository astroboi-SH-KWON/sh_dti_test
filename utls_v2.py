from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, f1_score, confusion_matrix
import sklearn.tree as tree
import pandas as pd
import numpy as np
import pickle
# np.random.seed(12)
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping
# from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, Activation, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.backend import set_session
import tensorflow.keras.backend as K
from multiprocessing import Pool
# import keras
import tensorflow as tf
pwd = '/home/ctilab/sh/test'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        

def tfidf_model_fit(df, feature, n_grams, max_features):
    stop_word_list = ['bbs', 'write', 'modify', 'board', 'delete', 'id', 'contents', 'writer', 'page']
    data = df.copy()
    data.fillna(' ', inplace = True)
    col_list = list(data.columns)
    fit_list = list(set(data[feature]))
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=n_grams, max_features = max_features, stop_words = stop_word_list) 
    tfidf_vectorizer.fit(fit_list)
    return tfidf_vectorizer

def tfidf_model_fit_char(df, char, n_grams, max_features, save_version, mode = 1):
    for i in char:
        feature = i

        if (mode == 1) & (feature == 'http_host') :
            max_features = 10
        elif (mode == 1) & (feature == 'http_agent') :
            max_features = 250
        elif (mode == 1) & (feature == 'http_query') :
            max_features = 2500

        tfidf_model = tfidf_model_fit(df, feature, n_grams, max_features)
        with open(pwd + "/model/"+str(feature)+"_tfidf_model_"+save_version+".pickle", "wb") as f:
            pickle.dump(tfidf_model, f)
        print(feature + " model save complite **************")

def tfidf_model_trans(model, df, feature, batch_size):
    data = df.copy()
    temp_batch = 0
    temp_df = pd.DataFrame()
    if len(data)%batch_size == 0:
        batch_count = int(len(data)/batch_size)
    else:
        batch_count = int(len(data)/batch_size) + 1
    for i in range(batch_count):
        if temp_batch + batch_size >= len(data):
            end_batch = len(data)
        else:
            end_batch = temp_batch + batch_size
        print(temp_batch, end_batch)
        trans_list = list(data[feature][temp_batch : end_batch])
        temp_batch += batch_size
        flag = True
        tries = 0
        while flag and tries < 10:
            try:
                tries += 1
                with Pool(20) as p:
                    tf_data = p.map(model.transform, [[item] for item in trans_list])
                    p.close()
                    p.join()
                flag = False
            except:
                pass
        tf_feature = model.get_feature_names()
        col_names = [feature + '_' + name for name in tf_feature]
        tf_df = pd.DataFrame(columns=col_names, data = np.concatenate([item.toarray() for item in tf_data]))
        temp_df = pd.concat([temp_df, tf_df], sort = True)
    temp_df.fillna(0, inplace = True)
    temp_df.reset_index(drop = True, inplace = True)
    return temp_df, tf_feature

def tfidf_model_trans_char(df, data_list, batch_size, save_version):
    data = df.copy()
    temp_df = pd.DataFrame(index = range(0, len(data)))
    str_data = df.copy()
    column_cnt = []
    for i in data_list:
        feature = i

        with open(pwd + "/model/"+str(feature)+"_tfidf_model_"+save_version+".pickle", "rb") as f:
            tfidf_model = pickle.load(f)
        
        print(feature + " model load complite **************")

        temp, _ = tfidf_model_trans(tfidf_model, data, feature, batch_size) ### max feature
        # str_data[i] = temp.apply(lambda x: tuple(x), axis = 1).apply(np.array)
        str_data[i] = [tuple(_) for n in range(len(str_data))]
        print(feature + " columns count :" +str(len(temp.columns)))
        temp_df = pd.merge(temp_df, temp, left_index = True, right_index = True, suffixes=('_x_'+feature, '_y_'+feature))
        column_cnt.append(len(temp.columns))
        print("merge columns :" + str(len(temp_df.columns)))
    columns = temp_df.columns
    dict_values = {}
    start = 0
    for i, end in enumerate(column_cnt):
        dict_values[data_list[i]] = list(columns[start:start+end])
        start = start+end
    data = pd.merge(data, temp_df, left_index = True, right_index = True)
    data.drop(data_list, axis = 1, inplace = True)
    return data, data.columns, str_data, dict_values

def scl_model_fit(data, save_version):
    scl_model = MinMaxScaler()
    scl_model.fit(data)
    with open(pwd + "/model/scl_model_"+save_version+".pickle", "wb") as f:
        pickle.dump(scl_model, f)

def scl_model_trans(data, save_version):
    with open(pwd + "/model/scl_model_"+save_version+".pickle", "rb") as f:
        scl_model = pickle.load(f)
    return scl_model.transform(data)

def dt_model_fit(train_x, train_y, max_depth, save_version):
    dt_model = DecisionTreeClassifier(max_depth = max_depth, random_state = 1)
    dt_model.fit(train_x, train_y)
    with open(pwd + "/model/dt_model_"+save_version+".pickle", "wb") as fw:
        pickle.dump(dt_model, fw)
    return dt_model

def get_dtree(save_version, feature_names, labels, train_x):
    with open(pwd + "/model/dt_model_"+save_version+".pickle", "rb") as f:
        tree = pickle.load(f)
    clf = tree
    labels_list = labels
    model_id = []
    logtime = []
    attack_nm = []
    value = []
    features = []
    node_num = []
    node_from = []
    node_to_left = []
    node_to_right = []
    level = []

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape = n_nodes, dtype = np.int64)
    is_leaves = np.zeros(shape = n_nodes, dtype = bool)
    stack = [(0,0)] # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]

        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    for i in range(n_nodes):
        if is_leaves[i]:
            label = np.argmax(clf.tree_.value[i], 1)
            attack_nm.append(labels_list[label[0]])
            features.append('-')
            value.append('-')
            node_num.append(i)
            node_to_left.append('-')
            node_to_right.append('-')
            level.append(node_depth[i])
        else:
            attack_nm.append('-')
            features.append(list(feature_names)[feature[i]])
            value.append(threshold[i])
            node_num.append(i)
            node_to_left.append(children_left[i])
            node_to_right.append(children_right[i])
            level.append(node_depth[i])

    return attack_nm, features, value, node_num, node_to_left, node_to_right, level

def get_parent(x, temp):
    parent = '-'
    if x != 0:
        if x in list(temp['node_to_left']):
            parent = temp.loc[temp['node_to_left'] == x, 'node_num']
        if x in list(temp['node_to_right']):
            parent = temp.loc[temp['node_to_right'] == x, 'node_num']
        # print(parent.iloc[0])
        return parent.iloc[0]


def cnn_model_fit(train_x, train_y, EPOCHS, BATCH_SIZE, model_name = None, save_version = None):
    
    input_dim = train_x.shape[1]
    cnn_model = Sequential([
        Reshape((input_dim, 1), input_shape=(input_dim, )),
        Conv1D(8, 10, strides=1, activation=None, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2, padding='same'),
        Conv1D(16, 10, strides=1, activation=None, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2, padding='same'),
        Conv1D(32, 10, strides=1, activation=None, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2, padding='same'),
        Conv1D(64, 10, strides=1, activation=None, padding='same'),
        BatchNormalization(),
        Activation('relu', name='activation_3'),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid', name='predictions')
    ])
    cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    early_stop = EarlyStopping(monitor = 'loss', patience = 2, verbose = 1, min_delta = 0.001)
    model_metrics = cnn_model.fit(train_x, train_y, epochs = EPOCHS, callbacks = [early_stop], batch_size = BATCH_SIZE, verbose = 1, shuffle = True)
    cnn_model.save(pwd + '/model/cnn_model_'+model_name+'_'+save_version+'.h5')
    K.clear_session()
    return model_metrics

def cnn_model_predict(test_x, batch_size = 16, model_name = None, save_version = None) :
    cnn_model = tf.keras.models.load_model(pwd + '/model/cnn_model_'+model_name+'_'+save_version+'.h5')
    predicted = cnn_model.predict(test_x, batch_size = batch_size)
    K.clear_session()
    return predicted, cnn_model
    
def dnn_model_fit(train_x, train_y, EPOCHS, BATCH_SIZE, model_name = None, save_version = None):
    
    input_dim = train_x.shape[1]
    dnn_model = Sequential([
        Dense(input_dim, activation = 'relu', input_shape = (input_dim,)),
        Dropout(0.5),
        Dense(1024, activation = 'relu'),
        Dropout(0.5),
        Dense(256, activation = 'relu'),
        Dropout(0.5),
        Dense(128, activation = 'relu'),
        Dense(1, activation = 'sigmoid')
    ])
    dnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    early_stop = EarlyStopping(monitor = 'loss', patience = 2, verbose = 1, min_delta = 0.001)
    model_metrics = dnn_model.fit(train_x, train_y, epochs = EPOCHS, callbacks = [early_stop], batch_size = BATCH_SIZE, verbose = 1, shuffle = True)
    dnn_model.save(pwd + '/model/dnn_model_'+model_name+'_'+save_version+'.h5')
    K.clear_session()
    return model_metrics

def dnn_model_predict(test_x, batch_size = 16, model_name = None, save_version = None) :
    dnn_model = tf.keras.models.load_model(pwd + '/model/dnn_model_'+model_name+'_'+save_version+'.h5')
    predicted = dnn_model.predict(test_x, batch_size = batch_size)
    K.clear_session()
    return predicted

def dt_result_print(x, y, model):
    pred_y = model.predict(x)
    temp_pred_y = pd.get_dummies(pred_y)
    temp_data_y = pd.get_dummies(y)

    for i in list(set(temp_pred_y)):
        print('*'*30)
        print('<', i, '>')
        acc = accuracy_score(temp_data_y[i], temp_pred_y[i])
        recall = recall_score(temp_data_y[i], temp_pred_y[i])
        precision = precision_score(temp_data_y[i], temp_pred_y[i])
        f1_sc = f1_score(temp_data_y[i], temp_pred_y[i])
        print('accuracy', acc)
        print('recall', recall)
        print('precision', precision)
        print('f1_score', f1_sc)
        print(confusion_matrix(temp_data_y[i], temp_pred_y[i], labels = [0,1]))  

    return pred_y

def ae_model_fit(train_x, EPOCHS, BATCH_SIZE, model_name = None, save_version = None):

    input_dim = train_x.shape[1]
    ae_model = Sequential([
        Dense(input_dim, activation = 'relu', input_shape = (input_dim,)),
        Dropout(0.5),
        Dense(1024, activation = 'relu'),
        Dropout(0.5),
        Dense(512, activation = 'relu'),
        Dropout(0.5),
        Dense(256, activation = 'relu'),
        Dropout(0.5),       
        Dense(512, activation = 'relu'),
        Dense(1024, activation = 'relu'),
        Dense(input_dim, activation = 'sigmoid')
    ])
    ae_model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    early_stop = EarlyStopping(monitor = 'loss', patience = 5, verbose = 1, min_delta = 0.001)
    model_metrics = ae_model.fit(train_x, train_x, epochs = EPOCHS, callbacks = [early_stop], batch_size = BATCH_SIZE, verbose = 1, shuffle = True)
    ae_model.save(pwd + '/model/ae_model_'+model_name+'_'+save_version+'.h5')
    K.clear_session()
    return model_metrics

def ae_model_predict(test_x, batch_size = 16, model_name = None, save_version = None) :
    ae_model = tf.keras.models.load_model(pwd + '/model/ae_model_'+model_name+'_'+save_version+'.h5')
    ae_pred = ae_model.predict(test_x, batch_size = batch_size)
    K.clear_session()
    ae_result = np.mean(np.power(test_x - ae_pred, 2), axis = 1)
    return ae_result

def mad_score(points, THRESHOLD, z_score, option = ['box', 'mad']):
    if option == 'box':
#         q1 = points.quantile(0.001)
#         q3 = points.quantile(0.999)
        q1 = points.quantile(0.25)
        q3 = points.quantile(0.75)
        iqr = q3 - q1
#         threshold_value = (q3 + 500 * iqr)
        threshold_value = (q3 + 26 * iqr)
        print('q1 :', q1)
        print('q3 :', q3)
        print('iqr :', iqr)
        print('threshold_value :', threshold_value)
        print('*'*30)
        return threshold_value
    else:
        median = np.median(points, axis = 0)
        deviation = np.abs(points - median)
        med_abs_deviation = np.median(deviation)
        modified_z_score = z_score * deviation / med_abs_deviation
        idx = (np.abs(modified_z_score - THRESHOLD)).argmin()
        threshold_value = points[idx]
        return modified_z_score, threshold_value

def def_att(x):
    if x == 'NORMAL':
        return 0
    elif x == 'CREDENTIAL':
        return 1
    elif x == 'SQL_INJECTION':
        return 2
    elif x == 'XSS':
        return 3
    elif x == 'file_download' or x == 'FILE DOWNLOAD':
        return 4
    elif x == 'ANOMALY':
        return 5