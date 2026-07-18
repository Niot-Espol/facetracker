"""Entrena y evalúa un clasificador multiclase de expresiones faciales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


SPLITS = ("Train", "Validation", "Test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/expressions"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-tflite", action="store_true")
    return parser.parse_args()


def dataset_path(root: Path, split: str) -> Path:
    path = root / split
    if not path.is_dir():
        raise FileNotFoundError(f"Falta la partición requerida: {path}")
    return path


def load_dataset(path: Path, image_size: int, batch_size: int, shuffle: bool, seed: int):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def build_model(image_size: int, num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="face_augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="expression")(x)
    return tf.keras.Model(inputs, outputs, name="expression_classifier"), base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")],
    )


def callbacks(output_dir: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "history.csv", append=True),
        tf.keras.callbacks.TensorBoard(log_dir=output_dir / "logs"),
    ]


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_dataset(dataset_path(args.data_dir, "Train"), args.image_size, args.batch_size, True, args.seed)
    validation = load_dataset(
        dataset_path(args.data_dir, "Validation"), args.image_size, args.batch_size, False, args.seed
    )
    test = load_dataset(dataset_path(args.data_dir, "Test"), args.image_size, args.batch_size, False, args.seed)
    class_names = train.class_names
    if validation.class_names != class_names or test.class_names != class_names:
        raise ValueError("Las clases de Train, Validation y Test deben coincidir exactamente.")

    autotune = tf.data.AUTOTUNE
    train = train.prefetch(autotune)
    validation = validation.prefetch(autotune)
    test = test.prefetch(autotune)

    model, base_model = build_model(args.image_size, len(class_names))
    compile_model(model, args.learning_rate)
    model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks(output_dir))

    if args.fine_tune_epochs:
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        compile_model(model, args.fine_tune_learning_rate)
        model.fit(
            train,
            validation_data=validation,
            initial_epoch=args.epochs,
            epochs=args.epochs + args.fine_tune_epochs,
            callbacks=callbacks(output_dir),
        )

    best_model = tf.keras.models.load_model(output_dir / "best_model.keras")
    test_metrics = {
        name: float(value) for name, value in best_model.evaluate(test, return_dict=True).items()
    }
    probabilities = best_model.predict(test)
    predicted = np.argmax(probabilities, axis=1)
    expected = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test], axis=0)
    confusion = tf.math.confusion_matrix(expected, predicted, num_classes=len(class_names)).numpy().tolist()
    summary = {"classes": class_names, "test_metrics": test_metrics, "confusion_matrix": confusion}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.export_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        (output_dir / "expression_classifier.tflite").write_bytes(converter.convert())

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
