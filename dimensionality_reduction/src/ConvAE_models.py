# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt

def load_and_normalize_data(data_dir='.../data/',
                            var='t2m'):
    """Load and normalize input fields (ensemble mean) from npy files. 
    All fields are normalized with field-specific min-max normalization 
    (i.e., with min and max computed individually for all fields)
    
    Parameters:
        data_dir: base directory where npy files are stored
        var: variable of interest
    Returns:
        x_train: normalized training data (2007-2014)
        x_valid: normalized validation data (2015)
        x_test: normalized test data
    All arrays are reshaped to (81,81,1) to serve as input for models later on
    """
    
    # load data
    fname = data_dir + 'gridded_ensmean_' + var + '.npy' 
    data_gridded = np.load(fname)

    # split into train, validation, test 
    train_end = 2920
    valid_start = 2920
    valid_end = 3285
    test_start = 3285

    x_train = data_gridded[:,:,:train_end] 
    x_train = np.moveaxis(x_train, 0, 1)

    mins_train = x_train.reshape(-1, x_train.shape[-1]).min(axis=0)
    maxs_train = x_train.reshape(-1, x_train.shape[-1]).max(axis=0)

    x_train_normalized = (x_train - mins_train)/(maxs_train - mins_train)
    x_train_normalized = x_train_normalized.reshape(81,81,1,x_train.shape[2])
    x_train_normalized = np.moveaxis(x_train_normalized, 3, 0)

    x_valid = data_gridded[:,:,valid_start:valid_end] 
    x_valid = np.moveaxis(x_valid, 0, 1)

    mins_valid = x_valid.reshape(-1, x_valid.shape[-1]).min(axis=0)
    maxs_valid = x_valid.reshape(-1, x_valid.shape[-1]).max(axis=0)

    x_valid_normalized = (x_valid - mins_valid)/(maxs_valid - mins_valid)
    x_valid_normalized = x_valid_normalized.reshape(81,81,1,x_valid.shape[2]) 
    x_valid_normalized = np.moveaxis(x_valid_normalized, 3, 0)

    x_test = data_gridded[:,:,test_start:] 
    x_test = np.moveaxis(x_test, 0, 1)

    mins_test = x_test.reshape(-1, x_test.shape[-1]).min(axis=0)
    maxs_test = x_test.reshape(-1, x_test.shape[-1]).max(axis=0)

    x_test_normalized = (x_test - mins_test)/(maxs_test - mins_test)
    x_test_normalized = x_test_normalized.reshape(81,81,1,x_test.shape[2]) 
    x_test_normalized = np.moveaxis(x_test_normalized, 3, 0)

    return (x_train_normalized, x_valid_normalized, x_test_normalized)


def fit_model(model,
              train_data,
              valid_data,
              max_epochs=100,
              es_patience=10,
              batch_size=32,
              verbose_fit = 1):
    """
    Fit function for autoencoder model
    
    Parameters:
        model: model to fit
        train_data: training data (from load_and_normalize_data)
        valid_data: validation data (from load_and_normalize_data)
        max_epochs: maximum number of epochs
        es_patience: patience for early stopping
    """
    
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   patience=es_patience, 
                                                   restore_best_weights = True)
    
    model.fit(train_data, train_data,
              epochs=max_epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=[es_callback],
              validation_data=(valid_data,valid_data),
              verbose = verbose_fit)
    

def plot_examples(model,
                  data,
                  inds,
                  plot_fname=None,
                  show = False,
                  save_fig = True):
    """
    Plot exemplary input, reconstruction and difference from training data
    
    Parameters:
        model: fitted model
        data: data with inds matching
        inds: indices referring to examples to compare
        show: display plot (True/False)
        save_fig: save figure (True/False)
        plot_fname: file name for figure if saved via save_fig
    """
    
    decoded_imgs = model.predict(data)
    
    columns = 3
    rows = inds.shape[0]
    
    fig=plt.figure(figsize=(4*columns, 4*rows))
    
    for i in range(0, rows):
        plt.subplot2grid((rows,columns), (i,0))
        plt.imshow(data[inds[i],:,:,0], vmin = 0, vmax = 1); plt.colorbar()
        plt.subplot2grid((rows,columns), (i,1))
        plt.imshow(decoded_imgs[inds[i],:,:,0], vmin = 0, vmax = 1); plt.colorbar()
        plt.subplot2grid((rows,columns), (i,2))
        plt.imshow(data[inds[i],:,:,0] - decoded_imgs[inds[i],:,:,0], 
                   cmap = 'RdBu', vmin = -1, vmax = 1); plt.colorbar()
    
    if show:
        plt.show()
        
    if save_fig:
        plt.savefig(plot_fname)
        

def plot_single_image_examples(model,
                  data,
                  inds,
                  plot_base_fname=None,
                  save_fig = True):
    """
    Plot exemplary input, reconstruction and difference from training data
    
    Parameters:
        model: fitted model
        data: data with inds matching
        inds: indices referring to examples to compare
        plot_base_fname: base name of plot file (folder/var_AE-dim_)
        show: display plot (True/False)
        save_fig: save figure (True/False)
        plot_fname: file name for figure if saved via save_fig
    """
    
    decoded_imgs = model.predict(data) 
    
    for i in range(0, inds.shape[0]):
        # original (= input) image
        plot_fname = plot_base_fname + str(inds[i]) + '_' + 'orig' + '.pdf'
        fig=plt.figure(figsize=(4, 4))
        plt.imshow(data[inds[i],:,:,0], vmin = 0, vmax = 1)
        plt.axis('off')
        if save_fig:
            plt.savefig(plot_fname,
                        bbox_inches='tight')
        plt.close(fig)
            
        # reconstructed (= output) image
        plot_fname = plot_base_fname + str(inds[i]) + '_' + 'reconstr' + '.pdf'
        fig=plt.figure(figsize=(4, 4))
        plt.imshow(decoded_imgs[inds[i],:,:,0], vmin = 0, vmax = 1)
        plt.axis('off')
        if save_fig:
            plt.savefig(plot_fname,
                        bbox_inches='tight')
        plt.close(fig)
        
        # difference
        plot_fname = plot_base_fname + str(inds[i]) + '_' + 'diff' + '.pdf'
        fig=plt.figure(figsize=(4, 4))
        plt.imshow(data[inds[i],:,:,0] - decoded_imgs[inds[i],:,:,0], 
                   cmap = 'RdBu', vmin = -1, vmax = 1)
        plt.axis('off')
        if save_fig:
            plt.savefig(plot_fname,
                        bbox_inches='tight')
        plt.close(fig)



def save_predictions(encoder_model,
                     filename=None,
                     save=False,
                     train_data=None,
                     valid_data=None,
                     test_data=None):
    
    preds_train = encoder_model.predict(train_data)
    preds_valid = encoder_model.predict(valid_data)
    preds_test = encoder_model.predict(test_data)
    
    tmp = np.append(preds_train, preds_valid, axis = 0)
    preds = np.append(tmp, preds_test, axis = 0)
    
    if save:
        np.save(arr=preds, file=filename)
        
    return (preds)


# ConvAE model with Conv2DTranspose in decoder part
# the implementation in the paper uses the 'simple' variant with min_size = 9
# learning_rate is not set here, defaults to 0.001 for Adam (could be specified as additional function argument)
def build_ConvAE_ConvTransp(encoding_dim,
                            min_size,
                            dec_kernel=6,
                            variant='simple',
                            output_activation='sigmoid',
                            optimizer = 'Adam',
                            loss = 'mean_squared_error'):
    
    """
    Build Convolutional Autoencoder, with a minimum dimension of min_size x min_size pixels
    Decoder uses Conv2DTranspose layers
    
    Parameters:
        variant: 'complex'/'simple', changes filter setup in Conv2D/Conv2DTranspose layers
        encoding_dim: dimension of encoded representation in latent space
        min_size: minimum size to which input field is downscaled in encoding part (3 = 3x3 or 9 = 9x9)
        dec_kernel: size of filter in Conv2DTranspose layers of decoder part (e.g., 3, 6, 9, ...)
        output_activation: activation function applied in output layer ('sigmoid'/'linear')
        optimizer: optimizer used to compile the model
        loss: loss function used to compile the model ('mean_squared_error'/'binary_crossentropy')
    Returns:
        autoencoder: full autoencoder model returning decoded image
        encoder: encoder part, returning the latent representation
    """
    # reset Keras models
    tf.compat.v1.reset_default_graph()
    keras.backend.clear_session()
    
    if min_size == 3:       
        if variant == 'complex':
            channels = [32,16,8]
        elif variant == 'simple':
            channels = [16,8,4]
    elif min_size == 9:
        if variant == 'complex':
            channels = [32,16,8]
        elif variant == 'simple':
            channels = [16,8,4]

    input_field = Input(shape=(81,81,1))

    x = Conv2D(channels[0], (3, 3), activation='relu', padding='same')(input_field)
    x = MaxPooling2D((3, 3), padding='same')(x)
    x = Conv2D(channels[1], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), padding='same')(x)
    x = Conv2D(channels[2], (3, 3), activation='relu', padding='same')(x)

    if min_size == 3:
        x = MaxPooling2D((3, 3), padding='same')(x)
    x = Flatten(data_format = 'channels_last')(x)
    encoded = Dense(encoding_dim, activation='linear')(x) 

    x = Dense(min_size*min_size*channels[2], activation='relu')(encoded)
    x = Reshape((min_size,min_size,channels[2]))(x)
    
    if min_size == 3:
        x = Conv2DTranspose(channels[2], (dec_kernel, dec_kernel), 
                            strides=(3,3), activation='relu', padding='same')(x) 
    if min_size == 9:
        x = Conv2DTranspose(channels[2], (dec_kernel, dec_kernel), 
                            strides=(1,1), activation='relu', padding='same')(x) 
    x = Conv2DTranspose(channels[1], (dec_kernel, dec_kernel), strides=(3,3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(channels[0], (dec_kernel, dec_kernel), strides=(3,3), activation='relu', padding='same')(x)

    decoded = Conv2D(1, (3, 3), activation=output_activation, padding='same')(x) 

    autoencoder = Model(input_field, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    encoder = Model(input_field, encoded)
    
    return(autoencoder, encoder)