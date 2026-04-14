import argparse
import os

import numpy as np
import tensorflow as tf


def load_class_names(labels_path):
    with open(labels_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def predict(model_path, img_path, labels_path=None):
    model = tf.keras.models.load_model(model_path)
    model.summary()

    if labels_path is None:
        labels_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "label.txt")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"Missing {labels_path!r}. Train with train.py (writes label.txt) or pass --labels."
        )
    label_type = load_class_names(labels_path)

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(255, 255))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    pred = int(np.argmax(model.predict(img_array, verbose=0)[0]))
    if pred >= len(label_type):
        raise ValueError(
            f"Model output index {pred} out of range for {len(label_type)} labels"
        )
    label = label_type[pred]
    print(label)
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict card condition from an image")
    parser.add_argument(
        "--model",
        default="my_model.h5",
        help="Path to trained .h5 model",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to image file",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="label.txt from training (default: next to --model)",
    )
    args = parser.parse_args()

    labels = args.labels.strip() or None
    predict(
        os.path.abspath(args.model),
        os.path.abspath(args.image),
        labels_path=os.path.abspath(labels) if labels else None,
    )
