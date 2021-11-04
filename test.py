import tensorflow as tf

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

if __name__ == "__main__":
    test(model_path, img_path)