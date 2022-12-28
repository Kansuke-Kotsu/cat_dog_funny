import streamlit as st
import time
import numpy as np
from PIL import Image
from keras.models import load_model
import cv2
import math
import random

# config
imsize = (64, 64)
testpic     = "dog1.jpeg"
keras_param1 = "model/cnn1.h5"
keras_param5 = "model/cnn5.h5"
keras_param20 = "model/cnn20.h5"

st.set_page_config(
    page_title="AIをだませ! Streamlitで画像認識",
    layout="wide",
    initial_sidebar_state="expanded"
)

# init
def init():
    if "init" not in st.session_state:
        st.session_state.init=True
        reset_session()
        count()
        return True
    else:
        return False

def layer_session(layer=0):
    st.session_state.layer=layer

def reset_session():
    st.session_state.now_tab=None
    layer_session()

def count():
    if "count" not in st.session_state:
        st.session_state.count=0
    st.session_state.count+=1

def back_btn():
    st.button(f"戻る",on_click=reset_session)
def pop_btn(label="pop",key=None, layer=0, onclick=lambda:None, done=None, description=None):
    placeholder=st.empty()
    with placeholder.container():
        if description:
            st.write(description)
        res=st.button(label,key=key,on_click=lambda:[placeholder.empty(),layer_session(layer),onclick()])
    if res:
        if done:
            with placeholder:
                done()
                placeholder.empty()

def load_image(path):
        img = Image.open(path)
        img = img.convert('RGB')
        # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
        img = img.resize(imsize)
        # 画像データをnumpy配列の形式に変更
        img = np.asarray(img)
        img = img / 255.0
        return img

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image
    
def rand_logic(param):
    rd = random.randint(100, 200)
    if rd%2==0:
        upload_image = st.file_uploader("猫っぽい犬の画像を選択しよう!", type="jpg")
        if upload_image:
            model = load_model(param)
            image = Image.open(upload_image)
            img_array = np.array(image)
            st.image(img_array, width=100)
            img = pil2cv(image)
            img = cv2.resize(img, (64, 64))
            # predict
            prd = model.predict(np.array([img]))
            print(prd) # 精度の表示
            prelabel = np.argmax(prd, axis=1)
            if prelabel == 0:
                st.write(math.floor(prd[0][0]*100), "すごい！君の勝ちだ！")
            elif prelabel == 1:
                st.write(math.floor(prd[0][1]*100), "残念、、AIを甘く見ちゃいけないよ")
    elif rd%2==1:
        upload_image = st.file_uploader("犬っぽい猫の画像を選択しよう!", type="jpg")
        if upload_image:
            model = load_model(param)
            image = Image.open(upload_image)
            img_array = np.array(image)
            st.image(img_array, width=100)
            img = pil2cv(image)
            img = cv2.resize(img, (64, 64))
            # predict
            prd = model.predict(np.array([img]))
            print(prd) # 精度の表示
            prelabel = np.argmax(prd, axis=1)
            if prelabel == 1:
                st.write(math.floor(prd[0][0]*100), "すごい！君の勝ちだ！")
            elif prelabel == 0:
                st.write(math.floor(prd[0][1]*100), "残念、、AIを甘く見ちゃいけないよ")   

# contents
st.title("AIをだませ!! <Dog or Cat?>")
def index():
    st.write("ここは **Indexページ** です。")
    pop_btn(label=f"初級:学習回数1回、正確性57%",layer=1,onclick=lambda:[count()])
    pop_btn(label=f"中級:学習回数5回、正確性71%",layer=2,onclick=lambda:[count()])
    pop_btn(label=f"上級:学習回数20回、正確性94%",layer=3,onclick=lambda:[count()])
def level1():
    st.write("初級")
    rd = random.randint(100, 200)
    if rd%2==0:
        upload_image = st.file_uploader("猫っぽい犬の画像を選択しよう!", type="jpg")
        if upload_image:
            model = load_model(keras_param1)
            image = Image.open(upload_image)
            img_array = np.array(image)
            st.image(img_array, width=100)
            img = pil2cv(image)
            img = cv2.resize(img, (64, 64))
            # predict
            prd = model.predict(np.array([img]))
            print(prd) # 精度の表示
            prelabel = np.argmax(prd, axis=1)
            if prelabel == 0:
                st.write(math.floor(prd[0][0]*100), "すごい！君の勝ちだ！")
            elif prelabel == 1:
                st.write(math.floor(prd[0][1]*100), "残念、、AIを甘く見ちゃいけないよ")
    elif rd%2==1:
        upload_image = st.file_uploader("犬っぽい猫の画像を選択しよう!", type="jpg")
        if upload_image:
            model = load_model(keras_param1)
            image = Image.open(upload_image)
            img_array = np.array(image)
            st.image(img_array, width=100)
            img = pil2cv(image)
            img = cv2.resize(img, (64, 64))
            # predict
            prd = model.predict(np.array([img]))
            print(prd) # 精度の表示
            prelabel = np.argmax(prd, axis=1)
            if prelabel == 1:
                st.write(math.floor(prd[0][0]*100), "すごい！君の勝ちだ！")
            elif prelabel == 0:
                st.write(math.floor(prd[0][1]*100), "残念、、AIを甘く見ちゃいけないよ")
    back_btn()
def level2():
    st.write("中級")
    rd = random.randint(100, 200)
    if rd%2==0:
        upload_image = st.file_uploader("猫っぽい犬の画像を選択しよう!", type="jpg")
        if upload_image:
            model = load_model(keras_param5)
            image = Image.open(upload_image)
            img_array = np.array(image)
            st.image(img_array, width=100)
            img = pil2cv(image)
            img = cv2.resize(img, (64, 64))
            # predict
            prd = model.predict(np.array([img]))
            print(prd) # 精度の表示
            prelabel = np.argmax(prd, axis=1)
            if prelabel == 0:
                st.write(math.floor(prd[0][0]*100), "すごい！君の勝ちだ！")
            elif prelabel == 1:
                st.write(math.floor(prd[0][1]*100), "残念、、AIを甘く見ちゃいけないよ")
    elif rd%2==1:
        upload_image = st.file_uploader("犬っぽい猫の画像を選択しよう!", type="jpg")
        if upload_image:
            model = load_model(keras_param5)
            image = Image.open(upload_image)
            img_array = np.array(image)
            st.image(img_array, width=100)
            img = pil2cv(image)
            img = cv2.resize(img, (64, 64))
            # predict
            prd = model.predict(np.array([img]))
            print(prd) # 精度の表示
            prelabel = np.argmax(prd, axis=1)
            if prelabel == 1:
                st.write(math.floor(prd[0][0]*100), "すごい！君の勝ちだ！")
            elif prelabel == 0:
                st.write(math.floor(prd[0][1]*100), "残念、、AIを甘く見ちゃいけないよ")
    back_btn()
def level3():
    st.write("上級")
    rd = random.randint(100, 200)
    if rd%2==0:
        upload_image = st.file_uploader("猫っぽい犬の画像を選択しよう!", type="jpg")
        if upload_image:
            model = load_model(keras_param20)
            image = Image.open(upload_image)
            img_array = np.array(image)
            st.image(img_array, width=100)
            img = pil2cv(image)
            img = cv2.resize(img, (64, 64))
            # predict
            prd = model.predict(np.array([img]))
            print(prd) # 精度の表示
            prelabel = np.argmax(prd, axis=1)
            if prelabel == 0:
                st.write(math.floor(prd[0][0]*100), "すごい！君の勝ちだ！")
            elif prelabel == 1:
                st.write(math.floor(prd[0][1]*100), "残念、、AIを甘く見ちゃいけないよ")
    elif rd%2==1:
        upload_image = st.file_uploader("犬っぽい猫の画像を選択しよう!", type="jpg")
        if upload_image:
            model = load_model(keras_param20)
            image = Image.open(upload_image)
            img_array = np.array(image)
            st.image(img_array, width=100)
            img = pil2cv(image)
            img = cv2.resize(img, (64, 64))
            # predict
            prd = model.predict(np.array([img]))
            print(prd) # 精度の表示
            prelabel = np.argmax(prd, axis=1)
            if prelabel == 1:
                st.write(math.floor(prd[0][0]*100), "すごい！君の勝ちだ！")
            elif prelabel == 0:
                st.write(math.floor(prd[0][1]*100), "残念、、AIを甘く見ちゃいけないよ")
    back_btn()
   
# body
init()
st.session_state.ck=0
# delay
time.sleep(0.1)
_layer=st.session_state.layer
if _layer==0: # index
    layer_session(1)
    index()
elif _layer==1: # 初級
    level1()
elif _layer==2: #中級
    level2()
elif _layer==3: #上級        
    level3()