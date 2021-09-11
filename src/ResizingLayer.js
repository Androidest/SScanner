import * as tf from '@tensorflow/tfjs';

// Define Custom Layer
export class Resizing extends tf.layers.Layer {
    constructor() {
        super({});
        this.target_width = 800
        this.target_height = 360
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.target_height, this.target_width, inputShape[3]]
    }

    call(inputs, kwargs) {
        this.invokeCallHook(inputs, kwargs);
        inputs = inputs[0]
        return tf.image.resizeNearestNeighbor(inputs, [this.target_height, this.target_width]);
    }

    static get className() {
        return 'Resizing';
    }
}
tf.serialization.registerClass(Resizing); // Needed for serialization.
