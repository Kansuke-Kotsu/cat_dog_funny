import numpy as np
import math
from PIL import Image
from keras.models import load_model
import streamlit as st
import cv2
import time


imsize = (64, 64)
testpic     = "dog1.jpeg"
keras_param1 = "model/cnn1.h5"
keras_param5 = "model/cnn5.h5"
keras_param20 = "model/cnn20.h5"

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

def main():
    st.title("AIをだませ!! <Dog or Cat?>")
    # upload Image2
    upload_image1 = st.file_uploader("初級「学習回数1回、正確性57%」", type="jpg")
    upload_image5 = st.file_uploader("初級「学習回数5回、正確性71%」", type="jpg")
    upload_image20 = st.file_uploader("上級「学習回数20回、正確性94%」", type="jpg")
    if upload_image1:
        model = load_model(keras_param1)
        image = Image.open(upload_image1)
        img_array = np.array(image)
        st.image(img_array, width=100)
        img = pil2cv(image)
        img = cv2.resize(img, (64, 64))

        # predict
        prd = model.predict(np.array([img]))
        print(prd) # 精度の表示
        prelabel = np.argmax(prd, axis=1)
        if prelabel == 0:
            st.write(math.floor(prd[0][0]*100), "%の確率で犬です！！")
        elif prelabel == 1:
            st.write(math.floor(prd[0][1]*100), "%の確率で猫です！！")
    if upload_image5:
        model = load_model(keras_param5)
        image = Image.open(upload_image5)
        img_array = np.array(image)
        st.image(img_array, width=100)
        img = pil2cv(image)
        img = cv2.resize(img, (64, 64))

        # predict
        prd = model.predict(np.array([img]))
        print(prd) # 精度の表示
        prelabel = np.argmax(prd, axis=1)
        if prelabel == 0:
            st.write(math.floor(prd[0][0]*100), "%の確率で犬です！！")
        elif prelabel == 1:
            st.write(math.floor(prd[0][1]*100), "%の確率で猫です！！")

    if upload_image20:
            model = load_model(keras_param20)
            image = Image.open(upload_image20)
            img_array = np.array(image)
            st.image(img_array, width=100)
            img = pil2cv(image)
            img = cv2.resize(img, (64, 64))

            # predict
            prd = model.predict(np.array([img]))
            print(prd) # 精度の表示
            prelabel = np.argmax(prd, axis=1)
            if prelabel == 0:
                st.write(math.floor(prd[0][0]*100), "%の確率で犬です！！")
            elif prelabel == 1:
                st.write(math.floor(prd[0][1]*100), "%の確率で猫です！！")
    


if __name__ == '__main__':
    main()