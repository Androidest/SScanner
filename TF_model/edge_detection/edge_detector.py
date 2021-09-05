# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, MobileNet, MobileNetV2, resnet50
from tensorflow.keras import layers
from tensorflow.python.keras.applications.resnet import ResNet50

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
def weighted_BCE(logits, labels):
    labels = tf.cast(labels, tf.float32)
    _, h, w, _ = labels.shape
    if h == None:
        h, w = 512, 512

    # compute pos weight mask from labels
    pos_mask = labels # 1 mask
    neg_mask = 1.0 - labels # 0 mask
    pos_count = tf.reduce_sum(pos_mask, axis=(1,2,3), keepdims=True) 
    neg_count = (h*w + 1) - pos_count 
    pos_weight = (pos_mask / pos_count) + (neg_mask / neg_count) # mask of norm weights
    tf.print(pos_weight.shape)

    # more numerical stable
    loss = bce(labels, logits, sample_weight=pos_weight)
    loss = tf.reduce_sum(loss, axis=(1,2)) 
    loss = tf.reduce_mean(loss)

    return loss

def showBaseSummary(base):
    base_model = base(
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
    resultImg = (p > 0)  * 255
    plt.figure(figsize = (10,10))
    plt.imshow(resultImg, cmap='gray')

def compile_model(input, output, lr):
    model = tf.keras.Model(inputs=input, outputs=output)
    opt_fn = tf.keras.optimizers.Adam(lr)
    # loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_fn = weighted_BCE
    model.compile(optimizer=opt_fn, loss=loss_fn)
    return model

def create_model(base, layer_index, lr=0.001):
    base_model = base(include_top=False, input_shape=(None, None, 3)) #TODO
    base_model.trainable = False # works only before compiling

    input = base_model.input
    hx = base_model.layers[layer_index].output  
    hx = create_block(hx)
    output = layers.Conv2D(filters=1, kernel_size=1, padding='same')(hx) # a non functional placeholder output layer

    model = compile_model(input, output, lr)
    return model

def insert_upConv(model, layer_index, lr=0.001):
    model.trainable = False # works only before compiling

    old_fused = model.layers[-1].input
    hx = model.layers[layer_index].output  
    hx = create_block(hx)
    fused = tf.concat([hx, old_fused], axis=-1)
    output = layers.Conv2D(filters=1, kernel_size=1, padding='same')(fused)

    new_model = compile_model(model.input, output, lr)
    return new_model

def create_block(hx):
    hx = layers.Conv2D(filters=1, kernel_size=1, padding='same')(hx)
    hx = layers.BatchNormalization()(hx)
    hx = layers.LeakyReLU(0.01)(hx)
    hx = layers.Resizing(512, 512, interpolation='nearest')(hx)
    
    hx = layers.Conv2D(filters=1, kernel_size=5, padding='same')(hx)
    hx = layers.BatchNormalization()(hx)
    hx = layers.LeakyReLU(0.01)(hx)
    hx = layers.Conv2D(filters=1, kernel_size=5, padding='same')(hx)
    hx = layers.BatchNormalization()(hx)
    hx = layers.LeakyReLU(0.01)(hx)

    return hx

# showBaseSummary(base=MobileNet)

# %%
import import_ipynb
from edge_data_generator import loadData, generate

# generate(w=512, h=512, start=0, numb=3000)
ds_train, _ = loadData(split_rate=0.3)

# ===== VGG16 =========
# model = create_model(base=VGG16, layer_index=13, lr=0.001)
# train_model(model, ds_train, initial_epoch=0, epochs=2, batchSize=32)
# model = insert_upConv(model, layer_index=9, lr=0.001)
# train_model(model, ds_train, initial_epoch=0, epochs=2, batchSize=32)
# model = insert_upConv(model, layer_index=5, lr=0.001)
# train_model(model, ds_train, initial_epoch=0, epochs=2, batchSize=32)
# model = insert_upConv(model, layer_index=2, lr=0.001)
# train_model(model, ds_train, initial_epoch=0, epochs=3, batchSize=32)
# model.trainable = True
# compile_model(model.input, model.output, lr=0.0001)
# train_model(model, ds_train, initial_epoch=0, epochs=10, batchSize=32)

# ===== Mobilenet V1 ======
model = create_model(base=MobileNet, layer_index=35, lr=0.01)
train_model(model, ds_train, initial_epoch=0, epochs=2, batchSize=4)
model = insert_upConv(model, layer_index=22, lr=0.001)
train_model(model, ds_train, initial_epoch=0, epochs=2, batchSize=4)
model = insert_upConv(model, layer_index=9, lr=0.001)
train_model(model, ds_train, initial_epoch=0, epochs=2, batchSize=4)

# ====== Mobilenet V2 ======
# model = create_model(base=MobileNetV2, layer_index=26, lr=0.001)
# train_model(model, ds_train, initial_epoch=0, epochs=1, batchSize=8)
# model = insert_upConv(model, layer_index=8, lr=0.001)
# train_model(model, ds_train, initial_epoch=0, epochs=5, batchSize=8)

# model = create_model(base=MobileNetV2, layer_index=53, lr=0.001)
# train_model(model, ds_train, initial_epoch=0, epochs=2, batchSize=32)
# model = insert_upConv(model, layer_index=26, lr=0.001)
# train_model(model, ds_train, initial_epoch=0, epochs=2, batchSize=32)
# model = insert_upConv(model, layer_index=8, lr=0.001)
# train_model(model, ds_train, initial_epoch=0, epochs=2, batchSize=32)
# model.trainable = True
# compile_model(model.input, model.output, lr=0.0001)
# train_model(model, ds_train, initial_epoch=0, epochs=10, batchSize=32)



#%%  Save Model & Test
for i in range(len(model.layers)):
    l = model.layers[i]
    if 'resizing' in l.name:
        l.target_height = 486
        l.target_width = 1080

filePath = './Models/edge_detector_MobileNetV1.h5'
model.save(filePath, overwrite=True, include_optimizer=False)
predict_test("./raw_dataset/test.jpg", maxSize=1080)


# %% Load Model & Test
import tensorflow as tf

def predict_test(filename, maxSize=1080):
    image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    image = tf.image.resize(image, (maxSize, maxSize), preserve_aspect_ratio=True)
    image = tf.cast(tf.reshape(image,(1)+image.shape), tf.float32) / 127.5 - 1
    p = model.predict(image)[0, :, :, 0]
    resultImg = (p > -0.3)  * 255
    plt.figure(figsize = (10,10))
    plt.imshow(resultImg, cmap='gray')

filePath = './Models/edge_detector_MobileNetV1.h5'
model = tf.keras.models.load_model(filePath)
predict_test("./raw_dataset/test.jpg", maxSize=1080)


