from keras.preprocessing.image import ImageDataGenerator
import scipy
from glob import glob
import numpy as np
import os
import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec

from keras.models import Model
from keras.layers import Input, Conv2D

#from __future__ import print_function, division

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dropout, Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.misc


from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


class DataLoader:
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):

        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_k_data(self, domain, image_number=10, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('%s/%s/*' % (self.dataset_name, data_type))

        images = np.array(path)
        if image_number < len(path):
            images = images[:image_number]

        imgs = []
        for img_path in images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs) / 127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False, aug=False):
        data_type = "train" if not is_testing else "test"
 
        path_A = glob('%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob('%s/%sB/*' % (self.dataset_name, data_type))

        if not is_testing and aug:
            datagen = ImageDataGenerator(
                zoom_range=0.2,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.5, 1.5]
            )

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                # Data augmenting
                if not is_testing and aug:
                    img_A = datagen.random_transform(img_A)
                    img_B = datagen.random_transform(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    
    
class ReflectionPadding2D(Layer):
    def __init__(self, padding, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3]
        )
        return shape

    def call(self, x, mask=None):
        width_pad, height_pad = self.padding
        return tf.pad(
            x,
            [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]],
            'REFLECT'
        )


nn_input = Input((128, 128, 3))
reflect_pad = ReflectionPadding2D(padding=(3, 3))(nn_input)
conv2d = Conv2D(32, kernel_size=7, strides=1, padding="valid")(reflect_pad)
model = Model(nn_input, conv2d)

class CycleGAN:
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'face_dataset'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle  # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

        self.current_epoch = 0

    def build_generator(self):
        def conv2d(layer_input, filters, f_size=4, stride=2, padding='valid'):
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding=padding)(layer_input)
            d = InstanceNormalization()(d)
            d = Activation('relu')(d)
            return d

        def res_block(layer_input, filters=256, use_dropout=False):
            y = ReflectionPadding2D(padding=(1, 1))(layer_input)
            y = conv2d(y, filters, 3, 1)
            if use_dropout:
                y = Dropout(0.5)(y)
            y = ReflectionPadding2D(padding=(1, 1))(y)
            y = conv2d(y, filters, 3, 1)
            return Add()([y, layer_input])

        def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
            u = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Activation('relu')(u)
            return u

        # Image input
        d0 = Input(shape=self.img_shape)
        d1 = ReflectionPadding2D(padding=(3, 3))(d0)
        # c7s1-64
        d1 = conv2d(d1, self.gf, 7, 1)
        # d128
        d2 = conv2d(d1, self.gf * 2, 3, padding='same')
        # d256
        d3 = conv2d(d2, self.gf * 4, 3, padding='same')
        # R256,R256,R256,R256,R256,R256
        r = res_block(d3)
        r = res_block(r)
        r = res_block(r)
        r = res_block(r)
        r = res_block(r)
        r = res_block(r)
        # u128
        u1 = deconv2d(r, self.gf * 2, 3)
        # u64
        u2 = deconv2d(u1, self.gf, 3)
        # c7s1-3
        u3 = ReflectionPadding2D(padding=(3, 3))(u2)
        output_img = Conv2D(self.channels, kernel_size=7, strides=1, activation='tanh')(u3)

        return Model(d0, output_img)

    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if normalization:
                d = InstanceNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        img = Input(shape=self.img_shape)
        # C64-C128-C256-C512
        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def load_network(self, n_epoch_start,path):
        # returns a compiled model identical to the previous one
        self.current_epoch = n_epoch_start+1
        self.combined.load_weights(path)

#    def train(self, epochs, batch_size=1, sample_interval=50):
#        start_time = datetime.datetime.now()
#        print(start_time)
#
#        # Adversarial loss ground truths
#        valid = np.ones((batch_size,) + self.disc_patch)
#        fake = np.zeros((batch_size,) + self.disc_patch)
#
#        for epoch in range(epochs):
#            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size, aug=True)):
#                # ----------------------
#                #  Train Discriminators
#                # ----------------------
#                # Translate images to opposite domain
#                fake_B = self.g_AB.predict(imgs_A)
#                fake_A = self.g_BA.predict(imgs_B)
#                # Train the discriminators (original images = real / translated = Fake)
#                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
#                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
#                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
#                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
#                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
#                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
#                # Total disciminator loss
#                d_loss = 0.5 * np.add(dA_loss, dB_loss)
#
#                # ------------------
#                #  Train Generators
#                # ------------------
#
#                # Train the generators
#                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
#                                                      [valid, valid,
#                                                       imgs_A, imgs_B,
#                                                       imgs_A, imgs_B])
#
#                elapsed_time = datetime.datetime.now() - start_time
#
#                # Plot the progress
#                print(
#                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
#                    % (epoch, epochs,
#                       batch_i, self.data_loader.n_batches,
#                       d_loss[0], 100 * d_loss[1],
#                       g_loss[0],
#                       np.mean(g_loss[1:3]),
#                       np.mean(g_loss[3:5]),
#                       np.mean(g_loss[5:6]),
#                       elapsed_time))
#
#                # If at save interval => save generated image samples
#                if batch_i % sample_interval == 0:
#                    self.sample_images(epoch+self.current_epoch, batch_i)
#
#            if epoch % 10 == 0 and (epoch != 0 or self.current_epoch != 0):
#                self.combined.save("models/combined_ep{}.h5".format(
#                    self.dataset_name,
#                    epoch+self.current_epoch)
#                )
#                self.combined.save(path+"models/combined_ep_{}.h5".format(
#                    epoch+self.current_epoch)
#                )

#    def sample_images(self, epoch, batch_i):
#        r, c = 2, 3
#        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
#        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)
#        # Translate images to the other domain
#        fake_B = self.g_AB.predict(imgs_A)
#        fake_A = self.g_BA.predict(imgs_B)
#        # Translate back to original domain
#        reconstr_A = self.g_BA.predict(fake_B)
#        reconstr_B = self.g_AB.predict(fake_A)
#
#        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
#        # Rescale images 0 - 1
#        gen_imgs = 0.5 * gen_imgs + 0.5
#
#        titles = ['Original', 'Translated', 'Reconstructed']
#        fig, axs = plt.subplots(r, c)
#        cnt = 0
#        for i in range(r):
#            for j in range(c):
#                axs[i, j].imshow(gen_imgs[cnt])
#                axs[i, j].set_title(titles[j])
#                axs[i, j].axis('off')
#                cnt += 1
#
#        fig.savefig("images/{}_paper/{}_{}.png".format(
#            self.dataset_name, epoch, batch_i))
#        fig.savefig(path+"images/{}_paper/{}_{}.png".format(
#            self.dataset_name, epoch, batch_i))
#        plt.close()

    def test_k_images(self, im_path="1.jpg",translated_path="t.jpg",output_path="o.jpg"):
        r, c = 1, 3
        
        img_res=(128, 128)

        
        imgs_A = scipy.misc.imread(im_path)
        imgs_A = scipy.misc.imresize(imgs_A, img_res)
        imgs_A = np.array(imgs_A)/127.5 - 1.
        
        
        imgs_A = np.expand_dims(imgs_A,axis=0)
        
        
        

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
#         fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
#         reconstr_B = self.g_AB.predict(fake_A)

        for image_i in range(len(imgs_A)):
            gen_imgs = np.concatenate([
                np.expand_dims(imgs_A[image_i], axis=0),
                np.expand_dims(fake_B[image_i], axis=0),
                np.expand_dims(reconstr_A[image_i], axis=0),
#                 np.expand_dims(imgs_B[image_i], axis=0),
#                 np.expand_dims(fake_A[image_i], axis=0),
#                 np.expand_dims(reconstr_B[image_i], axis=0)
            ])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
          
            translated_img = gen_imgs[1]
            scipy.misc.imsave(translated_path, translated_img)
             
          
          
          
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            cnt = 0
            for ax, title_im in zip(axs, ['original', 'translated', 'reconstructed']):
              ax.imshow(gen_imgs[cnt])
              ax.axis('off')
              ax.set_title(title_im.capitalize())
              ax.grid(True)
              cnt+=1
              
#            fig.savefig(output_path)
#            plt.show()
#            plt.close()
            
   
cyclegan = CycleGAN()
cyclegan.load_network(n_epoch_start = 220,path="models/combined_ep_221.h5")
         
app = Flask(__name__)


def model_predict(img_path):
    cyclegan.test_k_images(im_path=img_path)


       





@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'user_uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        model_predict(file_path)
#        print(preds)
    op_path = "output_image.jpg"
  
    return jsonify(result=op_path)


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()