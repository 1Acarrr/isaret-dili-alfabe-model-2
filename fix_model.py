import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

SEQUENCE_LEN = 30
NUM_FEATURES = 126
num_classes = 25 # We need to get this from the old model

# Eski modeli yükle
old_model = tf.keras.models.load_model(r"C:\Users\isaca\OneDrive\Desktop\model_training\output\best_model.keras")
num_classes = old_model.output_shape[-1]

print("Eski model yüklendi, num_classes:", num_classes)

inp = layers.Input(shape=(SEQUENCE_LEN, NUM_FEATURES), name="input")

# unroll=True ile yeni model
x = layers.LSTM(
    128,
    return_sequences=True,
    kernel_regularizer=regularizers.l2(1e-4),
    unroll=True,
    name="lstm_1"
)(inp)
x = layers.BatchNormalization(name="bn_lstm1")(x)
x = layers.Dropout(0.3, name="drop_lstm1")(x)

x = layers.LSTM(
    64,
    return_sequences=False,
    kernel_regularizer=regularizers.l2(1e-4),
    unroll=True,
    name="lstm_2"
)(x)
x = layers.BatchNormalization(name="bn_lstm2")(x)
x = layers.Dropout(0.3, name="drop_lstm2")(x)

x = layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
    name="dense_1"
)(x)
x = layers.BatchNormalization(name="bn_dense1")(x)
x = layers.Dropout(0.4, name="drop_dense1")(x)

x = layers.Dense(
    64,
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
    name="dense_2"
)(x)
x = layers.BatchNormalization(name="bn_dense2")(x)
x = layers.Dropout(0.3, name="drop_dense2")(x)

out = layers.Dense(
    num_classes,
    activation="softmax",
    name="output"
)(x)

new_model = models.Model(inputs=inp, outputs=out, name="sign_language_lstmdnn")

# Ağırlıkları kopyala
for layer in new_model.layers:
    old_layer = old_model.get_layer(layer.name)
    if old_layer is not None:
        layer.set_weights(old_layer.get_weights())

print("Ağırlıklar transfer edildi!")

new_model.save(r"C:\Users\isaca\OneDrive\Desktop\model_training\output\best_model.keras")
print("Yeni model best_model.keras üzerine yazıldı!")
