"""
CNN architectures for 96x96x3 binary tumor-patch classification.

Two options, both compiled the same way (binary_crossentropy, AUC as the
tracked metric since that's the competition's own scoring metric):

  build_cnn_from_scratch()
      A compact VGG-style stack: [Conv-Conv-Pool] x 4 with BatchNorm,
      GlobalAveragePooling instead of Flatten (fewer params, less
      overfitting risk with a dataset this size relative to a big FC
      head), Dropout, single sigmoid output.

  build_transfer_model(backbone)
      A pretrained ImageNet backbone (frozen initially) + a small
      classification head. On a task like this -- natural-ish RGB
      textures, hundreds of thousands of training images available in
      the real dataset -- transfer learning consistently outperforms a
      from-scratch CNN in published results on this exact competition,
      so this is the one worth reaching for if you have the real data
      and a GPU.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_from_scratch(
    image_size: int = 96,
    channels: int = 3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(image_size, image_size, channels))

    x = layers.Rescaling(1.0 / 255)(inputs)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="histopath_cnn_from_scratch")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def build_transfer_model(
    backbone: str = "mobilenet_v2",
    image_size: int = 96,
    channels: int = 3,
    learning_rate: float = 1e-4,
    freeze_backbone: bool = True,
) -> keras.Model:
    backbones = {
        "mobilenet_v2": keras.applications.MobileNetV2,
        "efficientnet_b0": keras.applications.EfficientNetB0,
        "resnet50": keras.applications.ResNet50,
    }
    if backbone not in backbones:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(backbones)}.")

    inputs = keras.Input(shape=(image_size, image_size, channels))

    base_model = backbones[backbone](
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, channels)
    )
    base_model.trainable = not freeze_backbone

    # Each Keras application has its own expected preprocessing; using the
    # matching one (rather than a generic /255 rescale) matters for
    # transfer-learning accuracy.
    preprocess_fn = {
        "mobilenet_v2": keras.applications.mobilenet_v2.preprocess_input,
        "efficientnet_b0": keras.applications.efficientnet.preprocess_input,
        "resnet50": keras.applications.resnet50.preprocess_input,
    }[backbone]

    x = layers.Lambda(preprocess_fn)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name=f"histopath_{backbone}_transfer")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model
