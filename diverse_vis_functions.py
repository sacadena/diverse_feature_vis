import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
slim = tf.contrib.slim
from collections import OrderedDict

import sys
pixelcnn_path = 'pixel-cnn/'

if pixelcnn_path not in sys.path:
    sys.path.append(pixelcnn_path)
    
from pixel_cnn_pp import model as pxpp_model, nn as pxpp_nn

def vgg19(images, reuse=False, pooling='max', subtract_mean=True, final_endpoint='pool5'):

    filter_size = [3, 3]
    conv1 = lambda net, name: slim.conv2d(net, 64, filter_size, padding='VALID', scope=name)
    conv2 = lambda net, name: slim.conv2d(net, 128, filter_size, padding='VALID', scope=name)
    conv3 = lambda net, name: slim.conv2d(net, 256, filter_size, padding='VALID', scope=name)
    conv4 = lambda net, name: slim.conv2d(net, 512, filter_size, padding='VALID', scope=name)
    conv5 = conv4
    pooling_fns = {'avg': slim.avg_pool2d, 'max': slim.max_pool2d}
    pool =  lambda net, name: pooling_fns[pooling](net, [2, 2], scope=name)
    dropout = lambda net, name: slim.dropout(net, 0.5, is_training=False, scope=name)

    layers = OrderedDict()
    layers['conv1/conv1_1'] = conv1
    layers['conv1/conv1_2'] = conv1
    layers['pool1'] = pool
    layers['conv2/conv2_1'] = conv2
    layers['conv2/conv2_2'] = conv2
    layers['pool2'] = pool
    layers['conv3/conv3_1'] = conv3
    layers['conv3/conv3_2'] = conv3
    layers['conv3/conv3_3'] = conv3
    layers['conv3/conv3_4'] = conv3
    layers['pool3'] = pool
    layers['conv4/conv4_1'] = conv4
    layers['conv4/conv4_2'] = conv4
    layers['conv4/conv4_3'] = conv4
    layers['conv4/conv4_4'] = conv4
    layers['pool4'] = pool
    layers['conv5/conv5_1'] = conv5
    layers['conv5/conv5_2'] = conv5
    layers['conv5/conv5_3'] = conv5
    layers['conv5/conv5_4'] = conv5
    layers['pool5'] = pool
    layers['fc6'] = lambda net, name: slim.conv2d(net, 4096, [7, 7], padding='VALID', scope=name)
    layers['dropout6'] =  dropout
    layers['fc7'] = lambda net, name: slim.conv2d(net, 4096, [1, 1], padding='VALID', scope=name)
    layers['dropout7'] =  dropout
    layers['fc8'] = lambda net, name: slim.conv2d(net, 1000, [1, 1], padding='VALID', scope=name)

    with tf.variable_scope('vgg_19', reuse=reuse) as sc:
        net = images
        if subtract_mean:
            net -= tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
        end_points = OrderedDict()
        with slim.arg_scope([slim.conv2d], trainable=False):
            for layer_name, layer_op in layers.items():
                last = final_endpoint == layer_name
                act_fn = tf.nn.relu if not last else None
                with slim.arg_scope([slim.conv2d], activation_fn=act_fn):
                    net = layer_op(net, layer_name)
                end_points[layer_name] = net
                if last:
                    break
    return end_points

RF_SIZES = OrderedDict()
RF_SIZES['conv1/conv1_1'] = 3
RF_SIZES['conv1/conv1_2'] = 5
RF_SIZES['pool1'] = 6
RF_SIZES['conv2/conv2_1'] = 10
RF_SIZES['conv2/conv2_2'] = 14
RF_SIZES['pool2'] = 16
RF_SIZES['conv3/conv3_1'] = 24
RF_SIZES['conv3/conv3_2'] = 32
RF_SIZES['conv3/conv3_3'] = 40
RF_SIZES['conv3/conv3_4'] = 48
RF_SIZES['pool3'] = 52
RF_SIZES['conv4/conv4_1'] = 68
RF_SIZES['conv4/conv4_2'] = 84
RF_SIZES['conv4/conv4_3'] = 100
RF_SIZES['conv4/conv4_4'] = 116
RF_SIZES['pool4'] = 124
RF_SIZES['conv5/conv5_1'] = 156
RF_SIZES['conv5/conv5_2'] = 188
RF_SIZES['conv5/conv5_3'] = 220
RF_SIZES['conv5/conv5_4'] = 252
RF_SIZES['pool5'] = 268

NORMS = OrderedDict()
NORMS['conv1/conv1_1'] = 321.44
NORMS['conv1/conv1_2'] = 538.35
NORMS['conv2/conv2_1'] = 1086.86
NORMS['conv2/conv2_2'] = 1531.58
NORMS['conv3/conv3_1'] = 2651.66
NORMS['conv3/conv3_2'] = 3553.22
NORMS['conv3/conv3_3'] = 4453.84
NORMS['conv3/conv3_4'] = 5362.63
NORMS['conv4/conv4_1'] = 7637.38
NORMS['conv4/conv4_2'] = 9473.19
NORMS['conv4/conv4_3'] = 11285.79
NORMS['conv4/conv4_4'] = 13161.68
NORMS['conv5/conv5_1'] = 17780.20
NORMS['conv5/conv5_2'] = 21605.78
NORMS['conv5/conv5_3'] = 25507.56
NORMS['conv5/conv5_4'] = 29358.54


VGG_CHECKPOINT_FILE = 'networks/vgg_normalized.ckpt'
PXPP_CHECKPOINT_FILE = 'networks/pxpp_cifar.ckpt'
#PXPP_CHECKPOINT_FILE = '/src/codebase/nets/pxpp_cifar.ckpt'


@tf.RegisterGradient("mygrad_sqrt")
def _mygrad(op, grad):
    # Spatial whitening by FFT assuming 1/sqrt(F) spectrum
    num_px = int(grad.shape[1])
    grad = tf.transpose(grad, [0, 3, 1, 2])
    grad_fft = tf.fft2d(tf.cast(grad, tf.complex64))
    t = np.minimum(np.arange(0, num_px), np.arange(num_px, 0, -1), dtype=np.float32)
    t = 1 / np.maximum(1.0, (t[None,:] ** 2 + t[:,None] ** 2) ** (1/4))
    F = tf.constant(t / t.mean(), dtype=tf.float32, name='F')
    grad_fft *= tf.cast(F, tf.complex64)
    grad = tf.ifft2d(grad_fft)
    grad = tf.transpose(tf.cast(grad, tf.float32), [0, 2, 3, 1])
    return grad

class SubspaceNet():
    def __init__(self,
                 layer_name,
                 target_direction,
                 num_images,
                 diversity,
                 naturalness,
                 feature_layer_name=None,
                 div_metric='euclidean',
                 target_metric='max',
                 input_size=None):
        
        assert layer_name in RF_SIZES, 'Invalid layer name: %s' % layer_name
        assert (feature_layer_name is None 
            or feature_layer_name in RF_SIZES), 'Invalid feature layer name: %s' % feature_layer_name
        if input_size is None:
            input_size = RF_SIZES[layer_name]
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            images_raw = tf.get_variable(
                'images',
                shape=[num_images, input_size, input_size, 3],
                initializer=tf.random_normal_initializer())
            norm = tf.sqrt(tf.reduce_sum(tf.square(images_raw), axis=[1, 2, 3], keepdims=True))
            self.images_raw = images_raw
            self.images = (NORMS[layer_name] / 2) * images_raw / norm
            
            # precondition gradient (only spatial whitening)
            with self.graph.gradient_override_map({"Identity": "mygrad_sqrt"}):
                self.images = tf.identity(self.images, name="Identity")
            
            self.vgg = vgg19(self.images, subtract_mean=False, final_endpoint=layer_name)
            vgg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
            saver_vgg = tf.train.Saver(var_list=vgg_vars)
            if type(target_direction) == int:
                unit_num = target_direction
                target_direction = np.zeros(int(self.vgg[layer_name].shape[-1]), dtype=np.float32)
                target_direction[unit_num] = 1
            else:
                if target_metric in ['max', 'cos']:
                    target_direction /= np.sqrt(np.sum(target_direction ** 2))
            self.target_direction = tf.constant(target_direction,
                                                shape=[target_direction.size],
                                                name='target_direction')
            vgg_flat = tf.reshape(self.vgg[layer_name], [num_images, -1])
            if target_metric in ['max', 'cos']:
                if target_metric == 'cos':
                    vgg_flat /= tf.sqrt(tf.reduce_sum(tf.square(vgg_flat), axis=1) + 0.01)[:,None]
                self.predictions = tf.tensordot(vgg_flat, self.target_direction,
                                                axes=[[1], [0]], name='predictions')
            else:
                self.predictions = -tf.reduce_mean(tf.square(vgg_flat - self.target_direction), axis=1)

            # diversity penalty
            if div_metric == 'euclidean':
                dist = lambda x, y: tf.sqrt(tf.reduce_mean(tf.square(x - y)))
            elif div_metric == 'cosine':
                dist = lambda x, y: 1 - tf.abs(tf.reduce_sum(x * y)) / (AVG_RF_SIZE ** 2)
            if num_images > 1:
                distances = []
                features = self.vgg[feature_layer_name] if feature_layer_name else self.images
                for i in range(num_images-1):
                    for j in range(i+1, num_images):
                        distances.append(dist(features[i,:,:,:], features[j,:,:,:]))
                self.min_distance = tf.reduce_min(tf.stack(distances))
                self.mean_distance = tf.reduce_mean(tf.stack(distances))
            else:
                self.min_distance = tf.constant(0.0, dtype=tf.float32)
                self.mean_distance = tf.constant(0.0, dtype=tf.float32)
                
            # natural image prior (PixelCNN++)
            if naturalness > 0:
                model = tf.make_template('model', pxpp_model.model_spec)
                # pad if input_size not divisible by 4
                input_size_pxpp = int(np.ceil(input_size / 4.0) * 4.0)
                pad1 = int(np.ceil((input_size_pxpp - input_size - 0.1) / 2.0))
                pad2 = int(np.floor((input_size_pxpp - input_size + 0.1) / 2.0))
                init_images = tf.placeholder(tf.float32, shape=[num_images, input_size_pxpp, input_size_pxpp, 3])
                self.pxpp_init = model(init_images, init=True)
                self.images_pxpp = tf.minimum(tf.maximum(self.images / 128, -1), 1)
                self.images_pxpp = tf.pad(self.images_pxpp, [[0, 0], [pad1, pad2], [pad1, pad2], [0, 0]])
                self.pxpp = model(self.images_pxpp)
                self.image_likelihood = pxpp_nn.discretized_mix_logistic_loss(self.images_pxpp, self.pxpp)
    
                pxpp_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
                saver_pxpp = tf.train.Saver(var_list=pxpp_vars)
                init_feed_dict = {init_images: np.zeros(init_images.shape)}
            else:
                self.image_likelihood = tf.constant(0.0)
                init_feed_dict = None

            self.diversity_loss = diversity * self.min_distance
            self.naturalness_loss = naturalness * self.image_likelihood
            
            self.loss = -tf.reduce_mean(self.predictions) \
                        - self.diversity_loss \
                        + self.naturalness_loss

            self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[images_raw])
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer(), 
                             feed_dict=init_feed_dict)
            saver_vgg.restore(self.session, VGG_CHECKPOINT_FILE)
            if naturalness > 0:
                saver_pxpp.restore(self.session, PXPP_CHECKPOINT_FILE)

    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
        except:
            pass

    def map_subspace(self, max_iter=1000, learning_rate=1.0, patience=100, callback=None, callback_every=100):
        loss = []
        min_loss = 1e10
        not_decreased = 0
        alpha = 0.9
        for i in range(max_iter):
            _, loss_i = self.session.run([self.train_step, self.loss],
                                         {self.learning_rate: learning_rate})
            loss.append(loss_i)
            loss_ema = alpha * loss_ema + (1 - alpha) * loss_i if i > 0 else loss_i
            if loss_ema < min_loss:
                min_loss = loss_ema
                not_decreased = 0
            else:
                not_decreased += 1
            if not_decreased > patience:
                callback(self, i)
                break
            if callback is not None and not ((i+1) % callback_every):
                callback(self, i)

        images, predictions, min_distance, mean_distance = self.session.run(
            [self.images, self.predictions, self.min_distance, self.mean_distance])
        return images, predictions, min_distance, mean_distance, np.array(loss)

def deprocess_image(img):
    return np.clip(img + 128, 0, 255).astype('uint8')

def callback(net, i):
    num_images = int(net.images.shape[0])
    image, pred, div, nat, loss = net.session.run([net.images, net.predictions, net.diversity_loss, net.naturalness_loss, net.loss])
    print('Iteration {:d} | Loss: {:.2f} | Diversity: {:.2f} | Naturalness: {:.2f} | Avg activation: {:.2f}\nActivations: {}'.format(
        i+1, loss, div, nat, pred.mean(), pred))
    fig, axes = plt.subplots(1, num_images, figsize=((num_images+1)*3, 2))
    if num_images > 1:
        for im, ax in zip(image, axes.flatten()):
            ax.imshow(deprocess_image(im))
    else:
        axes.imshow(deprocess_image(image[0]))
    plt.show()
    plt.pause(0.05)
    


class SubspaceNetShiftInvariant():
    def __init__(self,
                 layer_name,
                 target_direction,
                 naturalness,
                 norm_penalty,
                 weight_min=0,
                 target_metric='max'):
        
        assert layer_name in RF_SIZES, 'Invalid layer name: %s' % layer_name
        rf_size = RF_SIZES[layer_name]
        texture_size = 2 * rf_size - 1
        num_images = (rf_size - 1) ** 2

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.texture_raw = tf.get_variable(
                'texture_raw',
                shape=[1, texture_size, texture_size, 3],
                initializer=tf.random_normal_initializer())
            norm = lambda x: tf.sqrt(tf.reduce_sum(tf.square(x)))

            # mask
            t = (np.arange(rf_size) - (rf_size - 1) / 2).astype(np.float32)
            x, y = np.meshgrid(t, t)
            p = tf.get_variable('p_norm', shape=(), initializer=tf.constant_initializer(0.0))
            p_ = 2 + tf.abs(p)
            q = tf.get_variable('q_norm', shape=(), initializer=tf.constant_initializer(1.0))
            q_ = 1 + tf.abs(q)
            sigma = tf.get_variable('sigma', shape=(), initializer=tf.constant_initializer(np.log(rf_size)))
            sigma_ = tf.exp(sigma)
            r = (tf.abs(tf.constant(x)) ** p_ + tf.abs(tf.constant(y)) ** p_) ** (1 / p_)
            self.mask = tf.exp(-(tf.abs(r[None,...,None]) / sigma_) ** q_)
            self.mask_p = p_
            self.mask_q = q_
            self.mask_sigma = sigma_

            # crops with mask applied
            self.image_list = []
            self.norms = []
            for i in range(rf_size - 1):
                for j in range(rf_size - 1):
                    img = self.texture_raw[:,i:i+rf_size,j:j+rf_size,:] * self.mask
                    self.image_list.append(img)
                    self.norms.append(tf.reduce_sum(tf.square(img)))

            avg_norm = tf.sqrt(tf.reduce_mean(tf.stack(self.norms)))
            scale = (NORMS[layer_name] / 2) / avg_norm
            self.texture = scale * self.texture_raw
            self.images = scale * tf.concat(self.image_list, axis=0, name='image_batch')

            # precondition gradient (only spatial whitening)
            with self.graph.gradient_override_map({"Identity": "mygrad_sqrt"}):
                self.images = tf.identity(self.images, name="Identity")

            self.vgg = vgg19(self.images, subtract_mean=False, final_endpoint=layer_name)
            vgg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
            saver_vgg = tf.train.Saver(var_list=vgg_vars)
            if type(target_direction) == int:
                unit_num = target_direction
                target_direction = np.zeros(int(self.vgg[layer_name].shape[-1]), dtype=np.float32)
                target_direction[unit_num] = 1
            else:
                if target_metric in ['max', 'cos']:
                    target_direction /= np.sqrt(np.sum(target_direction ** 2))
            self.target_direction = tf.constant(target_direction,
                                                shape=[target_direction.size],
                                                name='target_direction')
            vgg_flat = tf.reshape(self.vgg[layer_name], [num_images, -1])
            if target_metric in ['max', 'cos']:
                if target_metric == 'cos':
                    vgg_flat /= tf.sqrt(tf.reduce_sum(tf.square(vgg_flat), axis=1) + 0.01)[:,None]
                self.predictions = tf.tensordot(vgg_flat, self.target_direction,
                                                axes=[[1], [0]], name='predictions')
            else:
                self.predictions = -tf.reduce_mean(tf.square(vgg_flat - self.target_direction), axis=1)

            # natural image prior (PixelCNN++)
            if naturalness > 0:
                model = tf.make_template('model', pxpp_model.model_spec)
                # pad if input_size not divisible by 4
                input_size_pxpp = int(np.ceil(texture_size / 4.0) * 4.0)
                pad1 = int(np.ceil((input_size_pxpp - texture_size - 0.1) / 2.0))
                pad2 = int(np.floor((input_size_pxpp - texture_size + 0.1) / 2.0))
                init_images = tf.placeholder(tf.float32, shape=[1, input_size_pxpp, input_size_pxpp, 3])
                self.pxpp_init = model(init_images, init=True)
                self.texture_pxpp = tf.minimum(tf.maximum(self.texture / 128, -1), 1)
                self.texture_pxpp = tf.pad(self.texture_pxpp, [[0, 0], [pad1, pad2], [pad1, pad2], [0, 0]])
                self.pxpp = model(self.texture_pxpp)
                self.image_likelihood = pxpp_nn.discretized_mix_logistic_loss(self.texture_pxpp, self.pxpp)
    
                pxpp_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
                saver_pxpp = tf.train.Saver(var_list=pxpp_vars)
                init_feed_dict = {init_images: np.zeros(init_images.shape)}
            else:
                self.image_likelihood = tf.constant(0.0)
                init_feed_dict = None

            self.naturalness_loss = naturalness * self.image_likelihood
            
            self.loss = - (1 - weight_min) * tf.reduce_mean(self.predictions) \
                        - weight_min * tf.reduce_min(self.predictions) \
                        + self.naturalness_loss

            self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
            var_list = [self.texture_raw, p, q, sigma]
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=var_list)
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer(), 
                             feed_dict=init_feed_dict)
            saver_vgg.restore(self.session, VGG_CHECKPOINT_FILE)
            if naturalness > 0:
                saver_pxpp.restore(self.session, PXPP_CHECKPOINT_FILE)

    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
        except:
            pass

    def map_texture(self, max_iter=1000, learning_rate=1.0, patience=100, callback=None, callback_every=100):
        loss = []
        min_loss = 1e10
        not_decreased = 0
        alpha = 0.9
        for i in range(max_iter):
            _, loss_i = self.session.run([self.train_step, self.loss],
                                         {self.learning_rate: learning_rate})
            loss.append(loss_i)
            loss_ema = alpha * loss_ema + (1 - alpha) * loss_i if i > 0 else loss_i
            if loss_ema < min_loss:
                min_loss = loss_ema
                not_decreased = 0
            else:
                not_decreased += 1
            if not_decreased > patience:
                break
            if callback is not None and not ((i+1) % callback_every):
                callback2(self, i)

        texture, mask, predictions, p, q, s = self.session.run([self.texture, self.mask, self.predictions, self.mask_p, self.mask_q, self.mask_sigma])
        return texture, mask, predictions, p, q, s, np.array(loss) 
    

def callback2(net, i):
    num_images = 6
    idx = np.round(np.linspace(0, len(net.image_list)-1, 6)).astype(np.int32)
    texture, mask, p, q, s, images, pred, nat, loss = net.session.run(
        [net.texture, net.mask, net.mask_p, net.mask_q, net.mask_sigma, net.images, net.predictions, net.naturalness_loss, net.loss])
    print('Iteration {:d} | Loss: {:.2f} | Naturalness: {:.2f} | Avg activation: {:.2f} | Min activations: {:.2f}'.format(
        i+1, loss, nat, pred.mean(), pred.min()))
    print('Mask: p = {:.2f}, q = {:.2f}, s = {:.2f}'.format(p, q, s))
    fig, axes = plt.subplots(1, num_images+2, figsize=((num_images+1)*3, 3))
    axes[0].imshow(deprocess_image(texture[0] * mask.max()))
    axes[1].imshow(mask[0,:,:,0])#, vmin=0, vmax=1)
    for i, ax in zip(idx, axes.flatten()[2:]):
        im = deprocess_image(images[i])
        ax.imshow(im)
    plt.show()
    plt.pause(0.05)


class SubspaceNetShiftInvariantFixed():
    def __init__(self,
                 layer_name,
                 target_direction,
                 naturalness,
                 norm_penalty,
                 weight_min=0,
                 target_metric='max'):
        
        assert layer_name in RF_SIZES, 'Invalid layer name: %s' % layer_name
        rf_size = RF_SIZES[layer_name]
        texture_size = 2 * rf_size - 1
        num_images = (rf_size - 1) ** 2

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.texture_raw = tf.get_variable(
                'texture_raw',
                shape=[1, texture_size, texture_size, 3],
                initializer=tf.random_normal_initializer())
            norm = lambda x: tf.sqrt(tf.reduce_sum(tf.square(x)))

            # mask
            t = (np.arange(rf_size) - (rf_size - 1) / 2).astype(np.float32)
            x, y = np.meshgrid(t, t)
            p_ = tf.constant(2.0)
            q_ = tf.constant(4.0)
            sigma_ = tf.constant(rf_size / 2.4) # 2.4 is a heuristic based on the ratio between conv3_1 rf size and 10 
            r = (tf.abs(tf.constant(x)) ** p_ + tf.abs(tf.constant(y)) ** p_) ** (1 / p_)
            self.mask = tf.exp(-(tf.abs(r[None,...,None]) / sigma_) ** q_)
            self.mask_p = p_
            self.mask_q = q_
            self.mask_sigma = sigma_

            # crops with mask applied
            self.image_list = []
            self.norms = []
            for i in range(rf_size - 1):
                for j in range(rf_size - 1):
                    img = self.texture_raw[:,i:i+rf_size,j:j+rf_size,:] * self.mask
                    self.image_list.append(img)
                    self.norms.append(tf.reduce_sum(tf.square(img)))

            avg_norm = tf.sqrt(tf.reduce_mean(tf.stack(self.norms)))
            scale = (NORMS[layer_name] / 2) / avg_norm
            self.texture = scale * self.texture_raw
            self.images = scale * tf.concat(self.image_list, axis=0, name='image_batch')

            # precondition gradient (only spatial whitening)
            with self.graph.gradient_override_map({"Identity": "mygrad_sqrt"}):
                self.images = tf.identity(self.images, name="Identity")

            self.vgg = vgg19(self.images, subtract_mean=False, final_endpoint=layer_name)
            vgg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
            saver_vgg = tf.train.Saver(var_list=vgg_vars)
            if type(target_direction) == int:
                unit_num = target_direction
                target_direction = np.zeros(int(self.vgg[layer_name].shape[-1]), dtype=np.float32)
                target_direction[unit_num] = 1
            else:
                if target_metric in ['max', 'cos']:
                    target_direction /= np.sqrt(np.sum(target_direction ** 2))
            self.target_direction = tf.constant(target_direction,
                                                shape=[target_direction.size],
                                                name='target_direction')
            vgg_flat = tf.reshape(self.vgg[layer_name], [num_images, -1])
            if target_metric in ['max', 'cos']:
                if target_metric == 'cos':
                    vgg_flat /= tf.sqrt(tf.reduce_sum(tf.square(vgg_flat), axis=1) + 0.01)[:,None]
                self.predictions = tf.tensordot(vgg_flat, self.target_direction,
                                                axes=[[1], [0]], name='predictions')
            else:
                self.predictions = -tf.reduce_mean(tf.square(vgg_flat - self.target_direction), axis=1)

            # natural image prior (PixelCNN++)
            if naturalness > 0:
                model = tf.make_template('model', pxpp_model.model_spec)
                # pad if input_size not divisible by 4
                input_size_pxpp = int(np.ceil(texture_size / 4.0) * 4.0)
                pad1 = int(np.ceil((input_size_pxpp - texture_size - 0.1) / 2.0))
                pad2 = int(np.floor((input_size_pxpp - texture_size + 0.1) / 2.0))
                init_images = tf.placeholder(tf.float32, shape=[1, input_size_pxpp, input_size_pxpp, 3])
                self.pxpp_init = model(init_images, init=True)
                self.texture_pxpp = tf.minimum(tf.maximum(self.texture / 128, -1), 1)
                self.texture_pxpp = tf.pad(self.texture_pxpp, [[0, 0], [pad1, pad2], [pad1, pad2], [0, 0]])
                self.pxpp = model(self.texture_pxpp)
                self.image_likelihood = pxpp_nn.discretized_mix_logistic_loss(self.texture_pxpp, self.pxpp)
    
                pxpp_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
                saver_pxpp = tf.train.Saver(var_list=pxpp_vars)
                init_feed_dict = {init_images: np.zeros(init_images.shape)}
            else:
                self.image_likelihood = tf.constant(0.0)
                init_feed_dict = None

            self.naturalness_loss = naturalness * self.image_likelihood
            
            self.loss = - (1 - weight_min) * tf.reduce_mean(self.predictions) \
                        - weight_min * tf.reduce_min(self.predictions) \
                        + self.naturalness_loss

            self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
            var_list = [self.texture_raw]
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=var_list)
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer(), 
                             feed_dict=init_feed_dict)
            saver_vgg.restore(self.session, VGG_CHECKPOINT_FILE)
            if naturalness > 0:
                saver_pxpp.restore(self.session, PXPP_CHECKPOINT_FILE)

    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
        except:
            pass

    def map_texture(self, max_iter=1000, learning_rate=1.0, patience=100, callback=None, callback_every=100):
        loss = []
        min_loss = 1e10
        not_decreased = 0
        alpha = 0.9
        for i in range(max_iter):
            _, loss_i = self.session.run([self.train_step, self.loss],
                                         {self.learning_rate: learning_rate})
            loss.append(loss_i)
            loss_ema = alpha * loss_ema + (1 - alpha) * loss_i if i > 0 else loss_i
            if loss_ema < min_loss:
                min_loss = loss_ema
                not_decreased = 0
            else:
                not_decreased += 1
            if not_decreased > patience:
                break
            if callback is not None and not ((i+1) % callback_every):
                callback2(self, i)

        texture, mask, predictions, p, q, s = self.session.run([self.texture, self.mask, self.predictions, self.mask_p, self.mask_q, self.mask_sigma])
        return texture, mask, predictions, p, q, s, np.array(loss) 
    
class Net():
    def __init__(self,
                 layer_name,
                 target_direction,
                 num_images,
                 normalize = False,
                 target_metric='max'):
        
        assert layer_name in RF_SIZES, 'Invalid layer name: %s' % layer_name
        #assert input_images.shape[1] == RF_SIZES[layer_name], 'Invalid input size for layer name: %s' % layer_name
        
        input_size = RF_SIZES[layer_name]
        
        self.graph = tf.Graph()
        with self.graph.as_default(): 
            self.images = tf.placeholder(tf.float32, shape=(num_images, input_size, input_size, 3), name='images')
            if normalize:
                norm = tf.sqrt(tf.reduce_sum(tf.square(self.images), axis=[1, 2, 3], keep_dims=True))
                self.images = (NORMS[layer_name]/2) * self.images / norm           
            self.vgg = vgg19(self.images, subtract_mean=False, final_endpoint=layer_name)
            vgg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
            saver_vgg = tf.train.Saver(var_list=vgg_vars)
            
            if type(target_direction) == int:
                unit_num = target_direction
                target_direction = np.zeros(int(self.vgg[layer_name].shape[-1]), dtype=np.float32)
                target_direction[unit_num] = 1
            else:
                if target_metric in ['max', 'cos']:
                    target_direction /= np.sqrt(np.sum(target_direction ** 2))
            self.target_direction = tf.constant(target_direction,
                                                shape=[target_direction.size],
                                                name='target_direction')            
            vgg_flat = tf.reshape(self.vgg[layer_name], [num_images, -1])
            if target_metric in ['max', 'cos']:
                if target_metric == 'cos':
                    vgg_flat /= tf.sqrt(tf.reduce_sum(tf.square(vgg_flat), axis=1) + tf.float32(0.01))[:,None]
                self.predictions = tf.tensordot(vgg_flat, self.target_direction,
                                                axes=[[1], [0]], name='predictions')
            else:
                self.predictions = -tf.reduce_mean(tf.square(vgg_flat - self.target_direction), axis=1)
            
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            saver_vgg.restore(self.session, VGG_CHECKPOINT_FILE)
            
    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
        except:
            pass
        
    def output(self, images):
        predictions = self.session.run([self.predictions],feed_dict={self.images:images})
        return predictions 
               
            













