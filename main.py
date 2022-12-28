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

# contents
st.title("AIをだませ!! <Dog or Cat?>")
def index():
    pop_btn(label=f"初級:普通にやっても2回に1回で誤認識します",layer=1,onclick=lambda:[count()])
    pop_btn(label=f"中級:ふつうに普通",layer=4,onclick=lambda:[count()])
    pop_btn(label=f"上級:こいつだませたら大したもん",layer=7,onclick=lambda:[count()])
    st.write("")
    st.write("遊び方")
    st.write("犬と猫を見分けるAIに、猫っぽい犬or犬っぽい猫の画像を見せて、バグらせよう!")
    st.write("レベルを選んで画像をアップ!")
def level1():
    st.write("初級「学習回数1回、正確性57%」")
    st.write("あなたが持っているのは・・・")
    pop_btn(label=f"猫っぽい犬の画像",layer=2,onclick=lambda:[count()])
    pop_btn(label=f"犬っぽい猫の画像",layer=3,onclick=lambda:[count()])
    back_btn()
def level2():
    st.write("中級「学習回数5回、正確性71%」")
    pop_btn(label=f"猫っぽい犬の画像",layer=5,onclick=lambda:[count()])
    pop_btn(label=f"犬っぽい猫の画像",layer=6,onclick=lambda:[count()])
    back_btn()
def level3():
    st.write("上級「学習回数20回、正確性94%」")
    pop_btn(label=f"猫っぽい犬の画像",layer=8,onclick=lambda:[count()])
    pop_btn(label=f"犬っぽい猫の画像",layer=9,onclick=lambda:[count()])
    back_btn()
    
def level1_c():
    st.write("初級「学習回数1回、正確性57%」")
    upload_image = st.file_uploader(label="猫っぽい犬", type="jpg")
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
        result = math.floor(prd[0][0]*100)
        st.write("犬:",result, "%","猫:",100 - result, "%")
        if 100-result > 90:
            st.write("完全に猫だ!頭がおかしくなりそうだよ!")
        elif 100-result >50:
            st.write("いい画像だ!確かに見る角度によっては猫だね!")
        else:
            st.write("残念、これは、、犬だ！")
    back_btn()
def level1_d():
    st.write("初級「学習回数1回、正確性57%」")
    upload_image = st.file_uploader(label="犬っぽい猫", type="jpg")
    if upload_image:
        model = load_model(keras_param1)
        image = Image.open(upload_image)
        img_array = np.array(image)
        st.image(img_array, width=100)
        img = pil2cv(image)
        img = cv2.resize(img, (64, 64))
        # predict
        prd = model.predict(np.array([img]))
        result = math.floor(prd[0][0]*100)
        st.write("犬:",result, "%","猫:",100 - result, "%")
        if result > 90:
            st.write("完全に犬だ!頭がおかしくなりそうだよ!")
        elif result >50:
            st.write("いい画像だ!確かに見る角度によっては犬だね!")
        else:
            st.write("残念、これは、、猫だ！")
    back_btn()

def level2_c():
    st.write("初級「学習回数5回、正確性71%」")
    upload_image = st.file_uploader(label="猫っぽい犬", type="jpg")
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
        result = math.floor(prd[0][0]*100)
        st.write("犬:",result, "%","猫:",100 - result, "%")
        if 100-result > 90:
            st.write("完全に猫だ!頭がおかしくなりそうだよ!")
        elif 100-result >50:
            st.write("いい画像だ!確かに見る角度によっては猫だね!")
        else:
            st.write("ん、これは、犬だよね？")
    back_btn()
def level2_d():
    st.write("初級「学習回数5回、正確性71%」")
    upload_image = st.file_uploader(label="犬っぽい猫", type="jpg")
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
        result = math.floor(prd[0][0]*100)
        st.write("犬:",result, "%","猫:",100 - result, "%")
        if result > 90:
            st.write("完全に犬だ!頭がおかしくなりそうだよ!")
        elif result >50:
            st.write("いい画像だ!確かに見る角度によっては犬だね!")
        else:
            st.write("犬っぽい画像を入れてみてくれるかな、？")
    back_btn()
  
def level3_c():
    st.write("初級「学習回数20回、正確性94%」")
    upload_image = st.file_uploader(label="猫っぽい犬", type="jpg")
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
        result = math.floor(prd[0][0]*100)
        st.write("犬:",result, "%","猫:",100 - result, "%")
        if 100-result > 90:
            st.write("完全に猫だ!頭がおかしくなりそうだよ!")
        elif 100-result >50:
            st.write("いい画像だ!確かに見る角度によっては猫だね!")
        else:
            st.write("これは、、犬だよね？")
    back_btn()
def level3_d():
    st.write("初級「学習回数20回、正確性94%」")
    upload_image = st.file_uploader(label="犬っぽい猫", type="jpg")
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
        result = math.floor(prd[0][0]*100)
        st.write("犬:",result, "%","猫:",100 - result, "%")
        if result > 90:
            st.write("完全に犬だ!頭がおかしくなりそうだよ!")
        elif result >50:
            st.write("いい画像だ!確かに見る角度によっては犬だね!")
        else:
            st.write("断然猫だね")
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
elif _layer==2: 
    level1_c()
elif _layer==3:       
    level1_d()
elif _layer==4: # 中級
    level2()
elif _layer==5:
    level2_c()
elif _layer==6: 
    level2_d()
elif _layer==7: #上級
    level3()
elif _layer==8:
    level3_c()
elif _layer==9:
    level3_d()