from symbol import parameters

import requests
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization


base_url="https://api.telegram.org/bot5668701212:AAG-LrOaB4YT7T6P5uYQKM2KbzNdnbTKDgM"

df = pd.read_csv('train.csv')
X = df['comment_text']
y = df[df.columns[2:]].values

MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,output_sequence_length=1800,output_mode='int')

vectorizer.adapt(X.values)
model = tf.keras.models.load_model('commenttoxicity.h5')

def del_msg(chat_id,msg_id):
    parameters={
        "chat_id":chat_id,"message_id":msg_id
    }

def read_msg(offset):
    parameters={
        "offset":offset
    }
    resp=requests.get(base_url+"/getupdates",data=parameters)
    data=resp.json()
    for result in data["result"]:
        inp=result["message"]["text"]
        print(inp)
        input_str = vectorizer(inp)
        res = model.predict(np.expand_dims(input_str,0))
        text = ''
        for idx, col in enumerate(df.columns[2:]):
            text += '{}: {}\n'.format(col, res[0][idx]>0.5)
        print(text)

    if data["result"]:
        return data["result"][-1]["update_id"]+1

offset=8
#del_msg()
while True:
    offset=read_msg(offset)

