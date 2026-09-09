"""
1D CNN for classifying clinical-literature text into one of 9 mutation
classes.

Input Text -> TextVectorization (int sequences) -> Embedding
  -> [Conv1D(kernel=k) -> GlobalMaxPooling1D] for k in (3, 4, 5)   (Kim, 2014 style)
  -> Concatenate -> Dropout -> Dense(128) -> Dropout -> Dense(9, softmax)

Multiple parallel kernel sizes let the network pick up short phrases
(kernel=3) and longer motifs (kernel=5) in the same pass, which tends to
beat a single-kernel-size CNN on this kind of variable-length scientific
text.

NOTE ON SUITABILITY: this architecture is included because the assignment
calls for a CNN. On this exact dataset, gradient-boosted trees over
TF-IDF/Word2Vec features and transformer encoders (e.g. BioBERT) have
outperformed CNN-based text classifiers in published writeups -- see the
README for citations and the included TF-IDF + Logistic Regression
baseline in src/baseline.py, which you should treat as the credibility
check on this model's log-loss, not as a discardable extra.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_text_cnn(
    vectorize_layer: layers.TextVectorization,
    vocab_size: int,
    num_classes: int = 9,
    embedding_dim: int = 128,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(1,), dtype="string")

    x = vectorize_layer(inputs)
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=False)(x)

    conv_blocks = []
    for kernel_size in (3, 4, 5):
        conv = layers.Conv1D(
            filters=128,
            kernel_size=kernel_size,
            activation="relu",
            padding="valid",
        )(x)
        pooled = layers.GlobalMaxPooling1D()(conv)
        conv_blocks.append(pooled)

    concatenated = layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    x = layers.Dropout(0.4)(concatenated)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="clinical_text_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
