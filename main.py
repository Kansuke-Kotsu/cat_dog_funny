import streamlit as st
import time
import numpy as np
from PIL import Image
from keras.models import load_model
import cv2

# config
st.set_page_config(
    page_title="Streamlitでのページ遷移とポップアップボタン",
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

# contents
def index():
    st.write("ここは **Indexページ** です。")
    pop_btn(label=f"初級",layer=1,onclick=lambda:[count()])
    pop_btn(label=f"中級",layer=2,onclick=lambda:[count()])
    pop_btn(label=f"上級",layer=3,onclick=lambda:[count()])
def level1():
    st.write("Level1です")
    back_btn()
def level2():
    st.write("Level2です")
    back_btn()
def level3():
    st.write("Level3です")
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