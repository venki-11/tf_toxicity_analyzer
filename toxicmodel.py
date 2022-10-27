import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization


df = pd.read_csv('train.csv')

X = df['comment_text']
y = df[df.columns[2:]].values

MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,output_sequence_length=1800,output_mode='int')
vectorizer.adapt(X.values)

model = tf.keras.models.load_model('commenttoxicity.h5')
input_str = vectorizer('hey i freaken hate you!')
res = model.predict(np.expand_dims(input_str,0))
print(res)
text=""
for idx, col in enumerate(df.columns[2:]):
    text += '{}: {}\n'.format(col, res[0][idx]>0.5)
print(text)
