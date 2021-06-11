import streamlit as st
import tensorflow as tf

@st.cache(suppress_st_warning=True)
def predict_food(image, model):
    img = load_image(image)
    preds = model.predict(img)
    highest_pred = tf.argmax(preds[0])
    return highest_pred

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

file = st.file_uploader(label="Throw your Images here:",
                        type=["jpg", "jpeg", "png"])

model = tf.keras.models.load_model("../models/B0_85_7.h5")

if not file:
    st.warning("Please upload an Image!")
    st.stop()
else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class_num = predict_food(image, model)
    pred_class_name = class_names[pred_class_num]
    st.write(pred_class_name)

    # Helpers --
    def load_image(image, image_shape=224):
        img = tf.image.decode_image(image, channels=3)
        img = tf.image.resize(image, size=([shape, shape]))
        img = tf.cast(tf.expand_dims(img, axis=0), tf.int16)
    r   eturn img



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