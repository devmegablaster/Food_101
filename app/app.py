import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt

def load_image(image, image_shape=224):
    img = tf.image.decode_image(image, channels=3)
    img = tf.image.resize(img, size=([image_shape, image_shape]))
    img = tf.cast(tf.expand_dims(img, axis=0), tf.int16)
    return img

class_names = ['apple_pie',
                'baby_back_ribs',
                'baklava',
                'beef_carpaccio',
                'beef_tartare',
                'beet_salad',
                'beignets',
                'bibimbap',
                'bread_pudding',
                'breakfast_burrito',
                'bruschetta',
                'caesar_salad',
                'cannoli',
                'caprese_salad',
                'carrot_cake',
                'ceviche',
                'cheesecake',
                'cheese_plate',
                'chicken_curry',
                'chicken_quesadilla',
                'chicken_wings',
                'chocolate_cake',
                'chocolate_mousse',
                'churros',
                'clam_chowder',
                'club_sandwich',
                'crab_cakes',
                'creme_brulee',
                'croque_madame',
                'cup_cakes',
                'deviled_eggs',
                'donuts',
                'dumplings',
                'edamame',
                'eggs_benedict',
                'escargots',
                'falafel',
                'filet_mignon',
                'fish_and_chips',
                'foie_gras',
                'french_fries',
                'french_onion_soup',
                'french_toast',
                'fried_calamari',
                'fried_rice',
                'frozen_yogurt',
                'garlic_bread',
                'gnocchi',
                'greek_salad',
                'grilled_cheese_sandwich',
                'grilled_salmon',
                'guacamole',
                'gyoza',
                'hamburger',
                'hot_and_sour_soup',
                'hot_dog',
                'huevos_rancheros',
                'hummus',
                'ice_cream',
                'lasagna',
                'lobster_bisque',
                'lobster_roll_sandwich',
                'macaroni_and_cheese',
                'macarons',
                'miso_soup',
                'mussels',
                'nachos',
                'omelette',
                'onion_rings',
                'oysters',
                'pad_thai',
                'paella',
                'pancakes',
                'panna_cotta',
                'peking_duck',
                'pho',
                'pizza',
                'pork_chop',
                'poutine',
                'prime_rib',
                'pulled_pork_sandwich',
                'ramen',
                'ravioli',
                'red_velvet_cake',
                'risotto',
                'samosa',
                'sashimi',
                'scallops',
                'seaweed_salad',
                'shrimp_and_grits',
                'spaghetti_bolognese',
                'spaghetti_carbonara',
                'spring_rolls',
                'steak',
                'strawberry_shortcake',
                'sushi',
                'tacos',
                'takoyaki',
                'tiramisu',
                'tuna_tartare',
                'waffles']
@st.cache(suppress_st_warning = True)
def predict_food(image, model):
    img = load_image(image)
    preds = model(img)
    highest_pred = tf.argmax(preds[0])
    highest_prob = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "Confidence": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('Confidence')
    return highest_pred, highest_prob, df

st.set_page_config(page_title="Food:101",
                   page_icon=":pizza:")

st.title("Food:101 :hamburger:")
st.markdown('''## Hey There! 
\n
Food 101 is a *Deep Learning Model* trained to predict what kind of food is present in an image
\n
As the name `Food 101` suggests, this model can predict ** 101 different food categories! **
\n
### Try out yourself!''')

st.sidebar.title("What's Food Vision ?")
st.sidebar.write("""
FoodVision is an end-to-end **Convolutional Neural Network Image Classification Model** which identifies the type of food in your image. 
\n
It can identify 101 different food classes.
\n
**Accuracy :** **`86%`**
\n
**Model :** **`EfficientNetB1`**
\n
**Training Duration :** **`117 minutes`**
\n
**Dataset :** **`Food101`**
\n
**Created By :** **Ojas Tyagi**
""")


file = st.file_uploader(label="Throw your Images here:",
                        type=["jpg", "jpeg", "png"])

model = tf.keras.models.load_model("./models/B0_85_7.h5")

if not file:
    st.warning("Please upload an Image!")
    st.stop()
else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class_num, prob, df = predict_food(image, model)
    pred_class_name = class_names[pred_class_num]
    pred_class_name = pred_class_name.capitalize()
    pred_class_name = pred_class_name.replace("_", " ")
    st.success(f"Prediction --> {pred_class_name} ({prob * 100} % Confidence)")
    st.write('## Top 5 Predictions -->')
    st.write(alt.Chart(df).mark_bar().encode(
        x='Confidence',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='Confidence'
    ).properties(width=600, height=400))

    # Helpers --
