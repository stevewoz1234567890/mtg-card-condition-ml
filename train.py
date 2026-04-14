import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import tensorflow as tf


def download_img(csv_path, download_path):
    data_path = pd.read_csv(csv_path)
    data_path = data_path.dropna(subset=["Woo image back", "Condition"])
    urls = data_path["Woo image back"].tolist()
    labels = data_path["Condition"]
    skus = data_path["SKU"] if "SKU" in data_path.columns else None
    label_type = labels.unique().tolist()

    os.makedirs(download_path, exist_ok=True)
    for label in label_type:
        os.makedirs(os.path.join(download_path, str(label)), exist_ok=True)

    for i, url in enumerate(urls):
        label = labels.iloc[i]
        base = url.split("/")[-1]
        if skus is not None:
            base = f"{skus.iloc[i]}_{base}"
        out_path = os.path.join(download_path, str(label), base)
        if os.path.isfile(out_path):
            continue
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(out_path, "wb") as handler:
                handler.write(resp.content)
            print("downloaded", url)
        except (requests.RequestException, OSError) as e:
            print("skip", url, ":", e)


def _preprocess_vgg(x, y):
    return tf.keras.applications.vgg16.preprocess_input(x), y


def split_train_test(img_path, test_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=img_path,
        validation_split=test_size,
        subset="training",
        seed=123,
        image_size=(255, 255),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=img_path,
        validation_split=test_size,
        subset="validation",
        seed=123,
        image_size=(255, 255),
        batch_size=batch_size,
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(_preprocess_vgg, num_parallel_calls=autotune)
    val_ds = val_ds.map(_preprocess_vgg, num_parallel_calls=autotune)
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds


def train(
    download=False,
    csv_path="",
    download_path="",
    img_path="",
    test_size=0.1,
    batch_size=32,
    epochs=1,
    model_out="my_model.h5",
    labels_out="label.txt",
    no_gpu=False,
):
    if no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if download:
        if not csv_path:
            raise ValueError("csv_path is required when download=True")
        download_img(csv_path, download_path)
        img_path = download_path

    if not img_path or not os.path.isdir(img_path):
        raise ValueError("img_path must be a directory of class subfolders (or use --download)")

    train_ds, val_ds = split_train_test(
        img_path=img_path, test_size=test_size, batch_size=batch_size
    )

    num_classes = len(train_ds.class_names)
    vgg = tf.keras.applications.VGG16(
        input_shape=(255, 255, 3), include_top=False, weights="imagenet"
    )
    vgg.trainable = False
    model = tf.keras.Sequential(
        [
            vgg,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs
    )

    model_path = os.path.abspath(model_out)
    model.save(model_path)

    labels_path = os.path.abspath(labels_out)
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_ds.class_names))

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(os.path.join(os.getcwd(), "loss.png"))
    plt.close()

    print("Saved model:", model_path)
    print("Saved class order:", labels_path)


def _default_csv():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "INVENTORY-TENSORFLOW_DATA - 5.0.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train card condition classifier (VGG16 head)")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download images from CSV URLs into --download-path",
    )
    parser.add_argument(
        "--csv-path",
        default=_default_csv(),
        help="CSV with columns: Woo image back, Condition (and optional SKU)",
    )
    parser.add_argument(
        "--download-path",
        default="training_data",
        help="Directory to store downloaded class subfolders",
    )
    parser.add_argument(
        "--img-path",
        default="",
        help="Image root with one subfolder per class (required if not --download)",
    )
    parser.add_argument("--test-size", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model-out", default="my_model.h5")
    parser.add_argument("--labels-out", default="label.txt")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Allow GPU (default: CPU only for reproducible small runs)",
    )
    args = parser.parse_args()

    img_path = args.img_path.strip()
    if not args.download and not img_path:
        candidate = os.path.abspath(args.download_path)
        if os.path.isdir(candidate):
            img_path = candidate
    img_path = os.path.abspath(img_path) if img_path else ""

    train(
        download=args.download,
        csv_path=args.csv_path,
        download_path=os.path.abspath(args.download_path),
        img_path=img_path,
        test_size=args.test_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_out=args.model_out,
        labels_out=args.labels_out,
        no_gpu=not args.gpu,
    )
