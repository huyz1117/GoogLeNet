import os
import tensorflow as tf

# Reference: https://github.com/wangtianrui/TfDatasetApiTest/blob/master/DataSetApiTest.py
def load_image(data_dir):
    all_classes = []
    all_images = []
    all_labels = []

    for i in os.listdir(data_dir):
        curren_dir = os.path.join(data_dir, i)
        if os.path.isdir(curren_dir):
            all_classes.append(i)
            for img in os.listdir(curren_dir):
                if img.endswith('png') or img.endswith('bmp') or img.endswith('jpg'):
                    all_images.append(os.path.join(curren_dir, img))
                    all_labels.append(all_classes.index(i))
        else:
            print(curren_dir, " doesnt exist")

    return all_classes, all_images, all_labels

# Reference: https://zhuanlan.zhihu.com/p/30751039
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # if comment the follow linem, the output shape can not be inferred
    image = tf.image.resize_images(image, [224, 224])
    image = tf.reshape(image, [224, 224, 3])
    image = (tf.cast(image, tf.float32) - 127.5)/127.5
    image = tf.image.random_flip_left_right(image)
    return image, label

# Reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/6_MultiGPU/multigpu_cnn.py
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


if __name__ == "__main__":
    data_dir = "/Users/apple/Desktop/dataset/mini-imagenet/train/"
    all_classes, all_images, all_labels = load_image(data_dir)
    print(len(all_classes))
    print(len(all_images))
    print(len(all_labels))
