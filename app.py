import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
from utils import load_and_prep, get_classes

@st.cache(suppress_st_warning=True)
def predicting(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df

class_names = get_classes()

st.set_page_config(page_title="Food-101",
                   page_icon="🍔")

#### SideBar ####

st.sidebar.title("What's Food-101 ?")
st.sidebar.write("""
Milestone Food-101 is an end-to-end **CNN Image Classification Model** which identifies the food in your image. 
It can identify over 100 different food classes
It is based upom a pre-trained Image Classification Model that comes with Keras and then retrained on the infamous **Food101 Dataset**.
**Accuracy :** **`82%`**
**Model :** **`EfficientNetB0`**
**Dataset :** **`Food101`**
""")


#### Main Body ####

st.title("Food Vision 🍔📷")
st.header("Identify what's in your food photos!")
st.write("To know more about this app, visit [**GitHub**](https://github.com/helloitsdaksh/Food-101)")
file = st.file_uploader(label="Upload an image of food.",
                        type=["jpg", "jpeg", "png"])


model = tf.keras.models.load_model("./model/EfficientNetB0.h5")


st.sidebar.markdown("Created by **Daksh Patel**")
st.sidebar.markdown(body="""
<th style="border:None"><a href="https://www.linkedin.com/in/daksh-patel-3a67101a3/" target="blank"><img align="center" src="https://bit.ly/3wCl82U" alt="dakshpatel" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://github.com/helloitsdaksh/" target="blank"><img align="center" src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="github" height="40"  /></a></th>
""", unsafe_allow_html=True)

if not file:
    st.warning("Please upload an image")
    st.stop()

else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class, pred_conf, df = predicting(image, model)
    st.success(f'Prediction : {pred_class} \nConfidence : {pred_conf*100:.2f}%')
    st.write(alt.Chart(df).mark_bar().encode(
        x='F1 Scores',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='F1 Scores'
    ).properties(width=600, height=400))
