# %%
import import_ipynb
import tensorflow as tf
from edge_data_generator import loadData

def create_model():
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False, input_shape=(None, None, 3))
    base_model.trainable = False
    base_model.summary()

    input = base_model.input
    hx = base_model.layers[26].output  # VGG:9, MobilenetV2:26,56
    hx = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(hx)
    output = tf.keras.layers.Conv2DTranspose( filters=1, kernel_size=4*2, strides=4, padding='same')(hx)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.summary()

    opt_fn = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(optimizer=opt_fn, loss=loss_fn)
    return model


def train_model(model, ds_train, ds_test=None, initial_epoch=0, epochs=50, batchSize=64):
    board_cb = tf.keras.callbacks.TensorBoard(
        log_dir='./tf_board',
        histogram_freq=0,  # How often to log histogram visualizations
        embeddings_freq=1,  # How often to log embedding visualizations
        update_freq="epoch",
    )

    final_epoch = initial_epoch+epochs
    for e in range(initial_epoch, final_epoch):
        x = ds_train.batch(batchSize)
        model.fit(x=x, epochs=e+1, initial_epoch=e,
                #   callbacks=[board_cb], # board_cb,
                  # validation_data=ds_test,
                  # validation_freq=1,
                  verbose=1)

        print('start training ...')


ds_train, _ = loadData(split_rate=0.2)
model = create_model()
model.trainable = False
train_model(model, ds_train, initial_epoch=0, epochs=5, batchSize=16)
model.trainable = True
train_model(model, ds_train, initial_epoch=0, epochs=15, batchSize=16)


# %%
import numpy as np
import matplotlib.pyplot as plt

# image_string = tf.io.read_file('./dataset/x/0.jpg')
image_string = tf.io.read_file('./raw_dataset/test.jpg')
image_decoded = tf.image.decode_jpeg(image_string, channels=3)
test = tf.image.resize(image_decoded, (1080, 1080), preserve_aspect_ratio=True)
test = tf.cast(test, tf.float32) / 127.5 - 1
test = tf.reshape(test, (1)+test.shape)
p = model.predict(test)[0, :, :, :]
p = p > 0.5
img = p  * 255
plt.figure(figsize = (9,12))
plt.imshow(img, cmap='gray')

np.max(img)

# %%
import tensorflow as tf
filePath = './Models/edge_detector_MobileNetV1.h5'
model.save(filePath, overwrite=True, include_optimizer=False)

# %%
import tensorflow as tf
filePath = './Models/edge_detector_MobileNetV1.h5'
model = tf.keras.models.load_model(filePath)


# %%
import tensorflow as tf
base_model = tf.keras.applications.mobilenet.MobileNet(
        include_top=False, input_shape=(512, 512, 3))
count =0
for l in base_model.layers:
    print(count, l.output.shape, l.name)
    count += 1