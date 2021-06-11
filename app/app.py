import streamlit as st

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

if not file:
    st.warning("Please upload an Image!")
    sr.stop()
else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")