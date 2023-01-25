from tkinter import *
from symbol import parameters
import tkinter
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization

ROOT = Tk()
ROOT.title("Toxicity analyzer")
ROOT.geometry('400x300')
LOOP_ACTIVE = True

base_url="https://api.telegram.org/bot5668701212:#######"

df = pd.read_csv('train.csv')
X = df['comment_text']
y = df[df.columns[2:]].values
rowi=0
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,output_sequence_length=1800,output_mode='int')

vectorizer.adapt(X.values)
model = tf.keras.models.load_model('commenttoxicity.h5')

def read_msg(offset):
    global rowi
    parameters={
        "offset":offset
    }
    resp=requests.get(base_url+"/getupdates",data=parameters)
    data=resp.json()
    for result in data["result"]:
        r=0
        inp=result["message"]["text"]
        sender=result["message"]["from"]["first_name"]
        print(inp)
        input_str = vectorizer(inp)
        res = model.predict(np.expand_dims(input_str,0))
        text = ''
        for idx, col in enumerate(df.columns[2:]):
            text += '{}: {}\n'.format(col, res[0][idx]>0.5)
            if(res[0][idx]>0.5):
                r=r+1
        if(r>2):
            s="The message/comment is toxic"
        else:
            s="the message/comment is intoxic"
        print(text)

        LABEL = Label(ROOT, text=text,borderwidth=3, relief="sunken",font=("Roboto slab",13))
        lbmsg=Label(ROOT,text="Message : "+inp+"\n"+"Sender : "+sender+"\n"+s,borderwidth=2,relief="solid",font=("Roboto slab",10))
        LABEL.grid(row=rowi,column=0,padx=5,pady=5,ipadx=2,ipady=2)
        lbmsg.grid(row=rowi,column=1,padx=7,pady=5,ipadx=2,ipady=2) 
        rowi=rowi+1
        print(rowi)
        print("N")
    if data["result"]:
        return data["result"][-1]["update_id"]+1
offset=842926556

while LOOP_ACTIVE:
    offset=read_msg(offset)
    ROOT.update()
    
