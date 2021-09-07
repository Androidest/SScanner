# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, MobileNet, MobileNetV2, resnet50
from tensorflow.keras import layers

weighted_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def weighted_BCE(labels, logits):
    # labels = tf.cast(labels, tf.float32)
    _, h, w, _ = labels.shape
    if h == None:
        h, w = 512, 512

    # compute pos weight mask from labels
    imgSize = h*w 
    pos_mask = labels # 1 mask
    neg_mask = 1.0 - labels # 0 mask
    pos_count = tf.reduce_sum(pos_mask, axis=(1,2,3), keepdims=True) 
    neg_count = imgSize - pos_count 
    pos_weight = tf.math.divide_no_nan(pos_mask,pos_count) + tf.math.divide_no_nan(neg_mask, neg_count) # mask of norm weights

    loss = weighted_bce(labels, logits, sample_weight=pos_weight) * imgSize

    return loss

def showBaseSummary(base):
    base_model = base(include_top=False, input_shape=(512, 512, 3))
    for i in range(len(base_model.layers)):
        l = base_model.layers[i]
        print(i, l.output.shape, l.name)

def train_model(model, ds_train, ds_test=None, epochs=10, batchSize=32):
    for e in range(0, epochs):
        x = ds_train.shuffle(2000).batch(batchSize)
        model.fit(x=x, epochs=e+1, initial_epoch=e, verbose=1
            #, validation_data=ds_test, validation_freq=1
        )

def fine_tune(model, lr, opt):
    model.trainable = True
    for layer in model.layers:
        if 'sscanner' not in layer.name:
            layer.trainable =  False
    
    return compile_model(model.input, model.output, lr, opt)

def predict_test(model, filename, maxSize=1080):
    image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    image = tf.image.resize(image, (maxSize, maxSize), preserve_aspect_ratio=True)
    image = tf.cast(tf.reshape(image,(1)+image.shape), tf.float32) / 127.5 - 1
    p = model.predict(image)[0, :, :, 0]
    resultImg = (p>0)  * 255
    plt.figure(figsize = (10,10))
    plt.imshow(resultImg, cmap='gray')

def compile_model(input, output, lr, opt="Adam"):
    model = tf.keras.Model(inputs=input, outputs=output)
    opt_fn = tf.keras.optimizers.Adam(lr)
    if opt == "SGD":
        print("Using SGD Optimizer")
        opt_fn = tf.keras.optimizers.SGD(lr)
    loss_fn = weighted_BCE
    model.compile(optimizer=opt_fn, loss=loss_fn)

    return model

def create_model(base, layer_index, scale, lr=0.001):
    base_model = base(include_top=False, input_shape=(None, None, 3)) #TODO
    base_model.trainable = False # works only before compiling

    input = base_model.input
    hx = base_model.layers[layer_index].output  
    hx = create_block(hx, scale)
    output = layers.Conv2D(filters=1, kernel_size=1, padding='same')(hx) # a non functional placeholder output layer

    model = compile_model(input, output, lr)
    return model

def insert_upConv(model, layer_index, scale, lr=0.001):
    model.trainable = False # works only before compiling

    old_fused = model.layers[-1].input
    hx = model.layers[layer_index].output  
    hx = create_block(hx, scale)
    fused = layers.Concatenate(axis=-1)([hx, old_fused])
    output = layers.Conv2D(filters=1, kernel_size=1, padding='same')(fused)

    new_model = compile_model(model.input, output, lr)
    return new_model

count = 0
def create_block(hx, scale):
    global count
    count += 1
    
    hx = layers.Conv2D(name="sscanner{i}_0".format(i=count), filters=8, kernel_size=1, padding='same')(hx)
    hx = layers.BatchNormalization()(hx)
    hx = layers.ReLU()(hx)

    hx = layers.Conv2DTranspose(name="sscanner{i}_1".format(i=count), filters=1, kernel_size=2*scale, strides=scale ,padding='same')(hx)

    hx = layers.Resizing(512, 512, interpolation='nearest')(hx)
    
    return hx

# showBaseSummary(base=MobileNetV2)

# %%
import import_ipynb
from edge_data_generator import loadData, generate

# generate(w=512, h=512, start=0, numb=3000, border_width=2)
ds_train, _ = loadData(split_rate=0.3)
ds_train = ds_train.prefetch(buffer_size=8)

# ===== VGG16 : too heavy for browser =========
    # model = create_model(base=VGG16, layer_index=13, lr=0.001)
    # train_model(model, ds_train, epochs=2, batchSize=32)
    # model = insert_upConv(model, layer_index=9, lr=0.001)
    # train_model(model, ds_train, epochs=2, batchSize=32)
    # model = insert_upConv(model, layer_index=5, lr=0.001)
    # train_model(model, ds_train, epochs=2, batchSize=32)
    # model = insert_upConv(model, layer_index=2, lr=0.001)
    # train_model(model, ds_train, epochs=3, batchSize=32)
    # model = fine_tune(model, lr=0.0001)
    # train_model(model, ds_train, epochs=7, batchSize=32)


# ===== Mobilenet V1 ======
# model = create_model(base=MobileNet, layer_index=72, scale=16, lr=0.001)
# train_model(model, ds_train, epochs=2, batchSize=8)
# model = insert_upConv(model, layer_index=35, scale=8, lr=0.001)
# train_model(model, ds_train, epochs=3, batchSize=8)
# model = insert_upConv(model, layer_index=22, scale=4, lr=0.001)
# train_model(model, ds_train, epochs=3, batchSize=8)
# model = insert_upConv(model, layer_index=9, scale=2, lr=0.001)
# train_model(model, ds_train, epochs=4, batchSize=8)
# ds_train, _ = loadData(split_rate=1)
# ds_train = ds_train.prefetch(buffer_size=32)
# model = fine_tune(model, lr=0.0001, opt="SGD")
# train_model(model, ds_train, epochs=5, batchSize=32)

# ====== Mobilenet V2 Small ======
model = create_model(base=MobileNetV2, layer_index=26, scale=4, lr=0.001) 
train_model(model, ds_train, epochs=3, batchSize=8)
model = insert_upConv(model, layer_index=8, scale=2, lr=0.001)
train_model(model, ds_train, epochs=5, batchSize=8)
ds_train, _ = loadData(split_rate=1)
ds_train = ds_train.prefetch(buffer_size=8)
model = fine_tune(model, lr=0.0001, opt="SGD")
train_model(model, ds_train, epochs=10, batchSize=8)

# ====== Mobilenet V2 Large ====== 115, 106, 97, 89, 80, 71
# model = create_model(base=MobileNetV2, layer_index=80, scale=16, lr=0.001)
# train_model(model, ds_train, epochs=2, batchSize=32)
# model = insert_upConv(model, layer_index=53, scale=8, lr=0.001)
# train_model(model, ds_train, epochs=3, batchSize=32)
# model = insert_upConv(model, layer_index=26, scale=4, lr=0.001)
# train_model(model, ds_train, epochs=3, batchSize=32)
# model = insert_upConv(model, layer_index=8, scale=2, lr=0.001)
# train_model(model, ds_train, epochs=4, batchSize=32)
# ds_train, _ = loadData(split_rate=0.3)
# ds_train = ds_train.prefetch(buffer_size=32)
# model = fine_tune(model, lr=0.0001, opt="SGD")
# train_model(model, ds_train, epochs=7, batchSize=32)



#%%  Save Model & Test
for i in range(len(model.layers)):
    l = model.layers[i]
    if 'resizing' in l.name:
        l.target_height = 486
        l.target_width = 1080

filePath = './Models/edge_detector_MobileNetV2.h5'
model.save(filePath, overwrite=True, include_optimizer=False)
predict_test(model, "./raw_dataset/test.jpg", maxSize=1080)


# %% Load Model & Test
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt

def predict_test(model, filename, maxSize=1080):
    image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    image = tf.image.resize(image, (maxSize, maxSize), preserve_aspect_ratio=True)
    image = tf.cast(tf.reshape(image,(1)+image.shape), tf.float32) / 127.5 - 1
    p = model.predict(image)[0, :, :, 0]
    resultImg = (p>0) * 255
    plt.figure(figsize = (10,10))
    plt.imshow(resultImg, cmap='gray')

filePath = './Models/edge_detector_MobileNetV2'
model = tf.keras.models.load_model(filePath+'.h5')
tfjs.converters.save_keras_model(model, filePath+"/")
predict_test(model, "./raw_dataset/test.jpg", maxSize=1080)
