# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Resizing

def class_balanced_sigmoid_cross_entropy(logits, label):
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y) # the number of 0 in y
    count_pos = tf.reduce_sum(y) # the number of 1 in y (less than count_neg)
    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits, y, pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta))

    return cost

def showBaseSummary():
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
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
    p = model.predict(image)[0, :, :, 0]
    resultImg = (p > 0.5)  * 255
    plt.figure(figsize = (10,10))
    plt.imshow(resultImg, cmap='gray')

def compile_model(input, output, lr):
    model = tf.keras.Model(inputs=input, outputs=output)
    opt_fn = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # loss_fn = class_balanced_sigmoid_cross_entropy
    model.compile(optimizer=opt_fn, loss=loss_fn)
    return model

def create_model(lr=0.001):
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2( 
        include_top=False, input_shape=(None, None, 3)) #TODO
    base_model.trainable = False # works only before compiling

    input = base_model.input

    hx = base_model.layers[26].output  # VGG:9, MobilenetV2:26,56
    hx = tf.keras.layers.Resizing(512, 512, interpolation='nearest')(hx)
    hx = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(hx)
    hx = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')(hx)
    hx = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')(hx)
    
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(hx) # a non functional placeholder output layer

    model = compile_model(input, output, lr)
    return model

def insert_upConv(model, layer_index, lr=0.01):
    model.trainable = False # works only before compiling
    old_fused = model.layers[-1].input

    hx = model.layers[layer_index].output  # VGG:9, MobilenetV2:26,56
    hx = tf.keras.layers.Resizing(512, 512, interpolation='nearest')(hx)
    hx = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(hx)
    hx = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')(hx)
    hx = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')(hx)
    
    fused = tf.concat([hx, old_fused], axis=-1)
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(fused)

    new_model = compile_model(model.input, output, lr)
    return new_model



# %%
import import_ipynb
from edge_data_generator import loadData, generate

# generate(w=512, h=512, start=0, numb=3000)

ds_train, _ = loadData(split_rate=1)
model = create_model(lr=0.001)
model = insert_upConv(model, layer_index=8, lr=0.001)
train_model(model, ds_train, initial_epoch=0, epochs=20, batchSize=16)
# model.trainable = True
# train_model(model, ds_train, initial_epoch=0, epochs=5, batchSize=16)


#%%
for i in range(len(model.layers)):
    l = model.layers[i]
    if 'resizing' in l.name:
        l.target_height = 486
        l.target_width = 1080

filePath = './Models/edge_detector_MobileNetV1.h5'
model.save(filePath, overwrite=True, include_optimizer=False)
predict_test("./raw_dataset/test.jpg", maxSize=1080)


# %%
import tensorflow as tf
filePath = './Models/edge_detector_MobileNetV1.h5'
model = tf.keras.models.load_model(filePath)
predict_test("./raw_dataset/test.jpg", maxSize=1080)


