# %%
import cv2
import numpy as np
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt


def loadData(split_rate=0.8):
    def parse_files(x_fname, y_fname):
        image_string = tf.io.read_file(x_fname)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        x = tf.cast(image_decoded, tf.float32) / 127.5 - 1

        image_string = tf.io.read_file(y_fname)
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        y = tf.cast(image_decoded, tf.float32) / 255.0
        
        return x, y

    x = tf.data.Dataset.list_files('./dataset/x/*.jpg', shuffle=False)
    x = list(x.as_numpy_iterator())

    y = tf.data.Dataset.list_files('./dataset/y/*.jpg', shuffle=False)
    y = list(y.as_numpy_iterator())

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=len(x), seed=123).map(parse_files)
    ds = ds.prefetch(buffer_size=8)
    print('data size: '+ str(len(x)))

    spliter = int(len(x) * split_rate)
    ds_train = ds.take(spliter)
    ds_test = ds.skip(spliter)
    return ds_train, ds_test


def create_model():
    base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(None, None, 3))
    base_model.trainable = False
    base_model.summary()

    input = base_model.input
    hx = base_model.layers[9].output
    output = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4*2, strides=4, padding='same')(hx)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.summary()

    opt_fn = tf.keras.optimizers.Adam(0.0001)
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
                # callbacks=[board_cb], # board_cb,
                # validation_data=ds_test, 
                # validation_freq=1,
                verbose=1)

        print('start training ...')



ds_train, _ = loadData(split_rate=0.1)
model = create_model()
model.trainable = False
train_model(model, ds_train, initial_epoch=0, epochs=1, batchSize=8)
model.trainable = True
train_model(model, ds_train, initial_epoch=0, epochs=10, batchSize=8)


# %%
# image_string = tf.io.read_file('./dataset/x/3000.jpg')
image_string = tf.io.read_file('./a.jpg')
image_decoded = tf.image.decode_jpeg(image_string, channels=3)
test = tf.image.resize(image_decoded, (1080,1080), preserve_aspect_ratio=True)
test = tf.cast(test, tf.float32) / 127.5 - 1
test = tf.reshape(test, (1)+test.shape)
img = (model.predict(test)[0,:,:,:] > 0) * 255
cv2.imwrite("./result.jpg", img)

np.max(img)

# %%
