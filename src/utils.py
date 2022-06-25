from tensorflow.keras.layers import Dense,BatchNormalization,Dropout,Embedding,Input,Concatenate,Reshape, Multiply, Subtract, Add, Multiply, Dropout, Subtract, Add,Lambda
from tensorflow.keras import Model
import numpy as np
# =============================================================================
# import tensorflow as tf
# import re
# =============================================================================
# =============================================================================
# serving = pd.read_csv('test.csv')
# serving.head()
# =============================================================================
from keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open('src/tokenizer/tokenizer.pickle', 'rb') as handle:
    fq = pickle.load(handle)

tokenizer = fq['foo']
word_index = tokenizer.word_index
pad_type = 'pre'
trunc_type = 'pre'

def cosine_distance(vests):
    
    """
    Calculate cosine distance
    """
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)






def build_model():
    
    """
    build siamese network 
    """
    input_1 = Input(shape = (14,))
# input_12 = Input(shape = (2,))
# input_2 = Input(shape = (12,))
# input_22 = Input(shape = (2,))

    x = Embedding( len(word_index) ,100, input_length = 14)(input_1)
    x = Reshape((1400,))(x)
    x = Dense(128,activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256,activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512,activation = 'relu')(x)


    embeddings = Model(inputs = [input_1], outputs = x)
    input_2 = Input(shape = (14,))
    input_3 = Input(shape = (14,))

    embeddings_1 = embeddings(input_2)
    embeddings_2 = embeddings(input_3)



    x3 = Subtract()([embeddings_1, embeddings_2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([embeddings_1, embeddings_1])
    x2_ = Multiply()([embeddings_2, embeddings_2])
    x4 = Subtract()([x1_, x2_])

        #https://stackoverflow.com/a/51003359/10650182
    x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([embeddings_1, embeddings_2])

    conc = Concatenate(axis=-1)([x5,x4, x3])

    x = Dense(100, activation="relu", name='conc_layer')(conc)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid", name = 'out')(x)

    model = Model([input_2, input_3], out)


    return model


def pre_process(X):
    """
    encode and concatenates incoming features
    """
    
    X = np.array(X).reshape(1,-1)
    lat, lon, category = X[:,0],X[:,1],X[:,2]
    seq =  tokenizer.texts_to_sequences(category)
    seq = pad_sequences(seq, padding=pad_type, truncating=trunc_type, maxlen=12)
    seq = np.concatenate([seq,np.reshape(lat,(-1,1))], axis = 1)
    seq = np.concatenate([seq,np.reshape(lon,(-1,1))], axis = 1)
    return seq
#pre_processed = pre_process(nm_df[nm_df['match']==1][['categories_1','latitude_1','longitude_1']].values)


def preprocess_doubelets(anchor, validation):
    """
    pre process
    """

    return (
        pre_process(anchor).astype('float32'),
        pre_process(validation).astype('float32')
      
    )

def predict(model,anchor, validate):
    
    """
    given an anchor and validate matches Point of interest
    """
# =============================================================================
#     match_list = []
#     for values in tqdm(df.iterrows()):
#         anchor, validate = preprocess_doubelets(np.array(values[1][1:4]).reshape(1,-1),np.array(values[1][5:]).reshape(1,-1))
#         pred = model.predict([ anchor, validate])
#         if pred > 0.3:
#             match = 1
#             
#         else :
#             match = 0
#         match_list.append(match)
#     df['match'] = match_list
# =============================================================================
    anchor, validate = preprocess_doubelets(anchor, validate)
    pred = model.predict([ anchor, validate])
    if pred >0.3:
        return 1
    else:
        return 0
