# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def showBaseSummary():
    base_model = tf.keras.applications.mobilenet.MobileNet(
                include_top=False, input_shape=(512, 512, 3))
    for i in range(len(base_model.layers)):
        l = base_model.layers[i]
        print(i, l.output.shape, l.name)

def train_model(model, ds_train, ds_test=None, initial_epoch=0, epochs=50, batchSize=64):
    final_epoch = initial_epoch+epochs
    for e in range(initial_epoch, final_epoch):
        x = ds_train.batch(batchSize)
        model.fit(x=x, epochs=e+1, initial_epoch=e, verbose=1
            #, validation_data=ds_test, validation_freq=1
        )

def predict_test(filename, maxSize=1080):
    image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    image = tf.image.resize(image, (maxSize, maxSize), preserve_aspect_ratio=True)
    image = tf.cast(tf.reshape(image,(1)+image.shape), tf.float32) / 127.5 - 1
    p = model.predict(image)[0, :, :, :]
    resultImg = (p > 0.5)  * 255
    plt.figure(figsize = (10,10))
    plt.imshow(resultImg, cmap='gray')

def create_model():
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2( include_top=False, input_shape=(None, None, 3))
    base_model.trainable = False
    showBaseSummary()

    input = base_model.input
    hx = base_model.layers[26].output  # VGG:9, MobilenetV2:26,56
    hx = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(hx)
    output = tf.keras.layers.Conv2DTranspose( filters=1, kernel_size=4*2, strides=4, padding='same')(hx)

    model = tf.keras.Model(inputs=input, outputs=output)
    opt_fn = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=opt_fn, loss=loss_fn)
    # model.summary()
    return model


# %%
import import_ipynb
from edge_data_generator import loadData

ds_train, _ = loadData(split_rate=0.2)
model = create_model()
model.trainable = False
train_model(model, ds_train, initial_epoch=0, epochs=5, batchSize=16)
model.trainable = True
train_model(model, ds_train, initial_epoch=0, epochs=15, batchSize=16)

filePath = './Models/edge_detector_MobileNetV1.h5'
model.save(filePath, overwrite=True, include_optimizer=False)
predict_test("./raw_dataset/test.jpg")


# %%
import tensorflow as tf
filePath = './Models/edge_detector_MobileNetV1.h5'
model = tf.keras.models.load_model(filePath)
predict_test("./raw_dataset/test.jpg")


