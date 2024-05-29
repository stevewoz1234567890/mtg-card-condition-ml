import tensorflow as tf
import numpy as np

def test(model_path, img_path):

    # Recreate the exact same model, including its weights and the optimizer
    new_model = tf.keras.models.load_model(model_path)

    # Show the model architecture
    new_model.summary()

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(255,255))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = np.argmax(new_model.predict(img_array)[0])

    label_type = ['DAMAGED', 'HP', 'LP', 'LP+', 'LP-', 'MP', 'MP+', 'MP-', 'NM']

    print(label_type[pred]) 
    return label_type[pred]

model_path = "C:\\Users\\ankur\\Documents\\GitHub\\clairvoyance\\purplemana_condition_classifer.h5"
#img_path = "C:\\Users\\ankur\\Documents\\GitHub\\clairvoyance\\examples\zz__raw-T473-1004-1-back-HP.jpg"
#img_path = "C:\\Users\\ankur\\Documents\\GitHub\\clairvoyance\\examples\zz__raw-TG-093021-A-back.jpg"
img_path = "C:\\Users\\ankur\\Documents\\GitHub\\clairvoyance\\examples\zz__raw-T425-0810-7-back.jpg"
#img_path = "C:\\Users\\ankur\\Documents\\GitHub\\clairvoyance\\examples\zz__raw-T413-0730-1-back.jpg"
#img_path = "C:\\Users\\ankur\\Documents\\GitHub\\clairvoyance\\examples\zz__T465-0922-8-back-deskew.jpg"



if __name__ == "__main__":
    test(model_path, img_path)
