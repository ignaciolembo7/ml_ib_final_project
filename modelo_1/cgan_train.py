# %%
#importo las librerias necesarias
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from keras.preprocessing import image
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization, Activation, Embedding, Concatenate
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# %%
noise_dim = 100
n_class = 10

tags = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

# Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

img_size = X_train.shape[1] # tamaño de las imagenes

X_train = np.reshape(X_train, [-1, img_size, img_size, 1])
X_train = np.repeat(X_train, 3, axis=-1)  # Convertir a 3 canales

# Normalize the data
X_train = (X_train - 127.5) / 127.5 # los valores se escalan para estar en el rango [-1, 1]

y_train = np.expand_dims(y_train, axis=-1) #expando la dimension de y_train para que quede analogo al ejemplo del cifar10

print(X_train.shape)
print(y_train.shape)

# %% [markdown]
# - Ploteo de un número random del dataset junto con su etiqueta

# %%
plt.figure(figsize=(2,2))
idx = np.random.randint(0,len(X_train))
img = image.array_to_img(X_train[idx], scale=True)
plt.imshow(img)
plt.axis('off')
plt.title(f"Etiqueta: {tags[y_train[idx][0]]}")
plt.show()

# %% [markdown]
# # Construcción del generador

# %%
def build_generator():
    # label input
    in_label = Input(shape=(1,), name='Label_Input')
    # create an embedding layer for all the 10 classes in the form of a vector of size 50 (ver paper)
    li = Embedding(n_class, 50, name='Embedding')(in_label)
 
    n_nodes = 7 * 7
    li = Dense(n_nodes, name='Label_Dense')(li)
    # reshape the layer
    li = Reshape((7, 7, 1), name='Label_Reshape')(li)
 
    # image generator input
    in_lat = Input(shape=(noise_dim,), name='Latent_Input')
 
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes, name='Generator_Dense')(in_lat)
    gen = LeakyReLU(negative_slope=0.2, name='Generator_LeakyReLU_1')(gen)
    gen = Reshape((7, 7, 128), name='Generator_Reshape')(gen)
 
    # merge image gen and label input
    merge = Concatenate(name='Concatenate')([gen, li])
 
    gen = Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same', name='Conv2DTranspose_1')(merge)  # 14x14x128
    gen = LeakyReLU(negative_slope=0.2, name='Generator_LeakyReLU_2')(gen)
 
    gen = Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same', name='Conv2DTranspose_2')(gen)  # 28x28x128
    gen = LeakyReLU(negative_slope=0.2, name='Generator_LeakyReLU_3')(gen)
 
    out_layer = Conv2D(
        3, (8, 8), activation='tanh', padding='same', name='Output_Conv2D')(gen)  # 28x28x3 #acá lo lleva a 32x32x3 porque es el tamaño de las imagenes del mnist 
 
    generator = Model([in_lat, in_label], out_layer, name='Generator')

    # Después, puedes usar plot_model para crear una imagen del modelo
    plot_model(generator, to_file='generator_structure.png', show_shapes=True, show_layer_names=True)

    return generator

# %%
g_optimizer=Adam(learning_rate=0.0002, beta_1 = 0.5) # Defino el optimizador
g_model = build_generator()
g_model.summary()

# %% [markdown]
# # Construcción del discriminador

# %%
def build_discriminator():
    
    # label input
    in_label = Input(shape=(1,), name='Label_Input')
    # This vector of size 50 will be learnt by the discriminator (ver paper)
    li = Embedding(n_class, 50, name='Embedding')(in_label)
   
    n_nodes = img_size * img_size 
    li = Dense(n_nodes, name='Label_Dense')(li) 
  
    li = Reshape((img_size, img_size, 1), name='Label_Reshape')(li) 
  
    # image input
    in_image = Input(shape=(img_size, img_size, 3), name='Image_Input') 
   
    merge = Concatenate(name='Concatenate')([in_image, li]) 
 
    # We will combine input label with input image and supply as inputs to the model. 
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='Conv2D_1')(merge) 
    fe = LeakyReLU(negative_slope=0.2, name='LeakyReLU_1')(fe)
   
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='Conv2D_2')(fe) 
    fe = LeakyReLU(negative_slope=0.2, name='LeakyReLU_2')(fe)
   
    fe = Flatten(name='Flatten')(fe) 
   
    fe = Dropout(0.4, name='Dropout')(fe)
   
    out_layer = Dense(1, activation='sigmoid', name='Output')(fe)
 
    # Define model the model. 
    discriminator = Model([in_image, in_label], out_layer, name='Discriminator')
    
       
    # Después, puedes usar plot_model para crear una imagen del modelo
    plot_model(discriminator, to_file='discriminator_structure.png', show_shapes=True, show_layer_names=True)

    return discriminator

# %%
d_optimizer=Adam(learning_rate=0.0002, beta_1 = 0.5) # Defino el optimizador
d_model = build_discriminator()
d_model.summary()

# %%
# Helper function to plot generated images
def show_samples(num_samples, n_class, g_model):
    fig, axes = plt.subplots(10,num_samples, figsize=(10,20)) 
    fig.tight_layout()
    fig.subplots_adjust(wspace=None, hspace=0.2)
 
    for l in np.arange(10):
      random_noise = tf.random.normal(shape=(num_samples, noise_dim))
      label = tf.ones(num_samples)*l
      gen_imgs = g_model.predict([random_noise, label])
      for j in range(gen_imgs.shape[0]):
        img = image.array_to_img(gen_imgs[j], scale=True)
        axes[l,j].imshow(img)
        axes[l,j].yaxis.set_ticks([])
        axes[l,j].xaxis.set_ticks([])
 
        if j ==0:
          axes[l,j].set_ylabel(tags[l])
    plt.show()

# %% [markdown]
# # Definición de las funciones de pérdida (loss)

# %%
# Define Loss function for Classification between Real and Fake
bce_loss = tf.keras.losses.BinaryCrossentropy()
 
# Discriminator Loss
def discriminator_loss(real, fake):
    real_loss = bce_loss(tf.ones_like(real), real) # Calculo la loss para las imagenes reales
    fake_loss = bce_loss(tf.zeros_like(fake), fake) # Calculo la loss para las imagenes falsas
    total_loss = real_loss + fake_loss
    return total_loss
   
# Generator Loss
def generator_loss(preds):
    return bce_loss(tf.ones_like(preds), preds) # Calculo la loss para el generador

# %% [markdown]
# # Paso de entrenamiento por batches 

# %%
@tf.function # Compiles the train_step function into a callable TensorFlow graph
def train_step(dataset, batch_size):
    
    real_images, real_labels = dataset # 
    # Sample random points in the latent space and concatenate the labels.
    random_latent_vectors = tf.random.normal(shape=(batch_size, noise_dim)) # Genero ruido aleatorio
    generated_images = g_model([random_latent_vectors, real_labels]) #  Genero imagenes falsas
 
    # Train the discriminator.
    with tf.GradientTape() as tape:
        pred_fake = d_model([generated_images, real_labels]) # Obtengo las predicciones del discriminador para las imagenes falsas
        pred_real = d_model([real_images, real_labels]) # Obtengo las predicciones del discriminador para las imagenes reales
         
        d_loss = discriminator_loss(pred_real, pred_fake) # Calculo la loss del discriminador
       
    grads = tape.gradient(d_loss, d_model.trainable_variables) # Calculo los gradientes
    d_optimizer.apply_gradients(zip(grads, d_model.trainable_variables)) # Aplico los gradientes al optimizador del discriminador
 
    #-----------------------------------------------------------------#
     
    # Sample random points in the latent space.
    random_latent_vectors = tf.random.normal(shape=(batch_size, noise_dim)) # Genero ruido aleatorio
    
    # Train the generator
    with tf.GradientTape() as tape: 
        fake_images = g_model([random_latent_vectors, real_labels]) # Genero imagenes falsas
        predictions = d_model([fake_images, real_labels]) # Obtengo las predicciones del discriminador para las imagenes falsas
        g_loss = generator_loss(predictions) # Calculo la loss del generador
     
    grads = tape.gradient(g_loss, g_model.trainable_variables) # Calculo los gradientes
    g_optimizer.apply_gradients(zip(grads, g_model.trainable_variables)) # Aplico los gradientes al optimizador del generador
    
    return d_loss, g_loss

# %%
def train(dataset, epoch_count, batch_size):

    d_loss_list_epoch = []
    g_loss_list_epoch = []
 
    for epoch in range(epoch_count):
        print('Epoch: ', epoch+1)
        d_loss_list_batch = []
        g_loss_list_batch = []
        q_loss_list = []
        start = time.time()
         
        itern = 0
        for image_batch in tqdm(dataset, desc=f"Número de batche: "): # Itero sobre todos los batches
            d_loss, g_loss = train_step(image_batch, batch_size) # Entreno el modelo
            d_loss_list_batch.append(d_loss) # Guardo la loss del discriminador
            g_loss_list_batch.append(g_loss) # Guardo la loss del generador
            itern=itern+1 
                 
        #show_samples(3, n_class, g_model)

        d_loss_list_epoch.append(np.mean(d_loss_list_batch))
        g_loss_list_epoch.append(np.mean(g_loss_list_batch))

        print (f'Epoch: {epoch+1} -- Generator Loss: {np.mean(g_loss_list_batch)}, Discriminator Loss: {np.mean(d_loss_list_batch)}\n')
        print (f'Took {time.time()-start} seconds. \n\n')

    return g_loss_list_epoch, d_loss_list_epoch

# %% [markdown]
# # Entrenamiento de la CGAN

# %%
epoch_count = 1 #100 # Cantidad de epocas 
batch_size = 16 #tamaño del batch hay 60000/16 = 3750 batches

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) # Se crea un dataset con los datos de entrenamiento
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size) # Se mezclan los datos del dataset cada 1000 y se agrupan en batches de a 16

g_loss_list, d_loss_list = train(dataset, epoch_count, batch_size)

g_model.save("gmodel_mnist_v1.keras")

# %% [markdown]
# # Funciones de loss para el generador y el discriminador

# %%
plt.figure(figsize=(8,6))
epochs = np.arange(1, epoch_count+1)
plt.plot(epochs, g_loss_list, label='Generator Loss')
plt.plot(epochs, d_loss_list, label='Discriminator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# # Generación de un dígito pedido por el usuario

# %%
# Cargar el modelo generador
g_model = load_model('gmodel_mnist_v1.keras')

# Preparar la etiqueta para el número 7
numero_a_generar = 7

label = np.expand_dims(numero_a_generar, axis=-1) #expando la dimension de y_train para que quede analogo al ejemplo del cifar10

# Generar ruido aleatorio
noise = np.random.normal(size=(1, noise_dim))

# Generar imagen falsa 
generated_image = g_model([noise, label]) #  Genero imagenes falsas
#generated_image = g_model.predict([noise, label])

plt.figure(figsize=(2,2))
print(generated_image.shape)
img = image.array_to_img(generated_image[0], scale=True)
plt.imshow(img)
plt.axis('off')
plt.title(f"{numero_a_generar}")
plt.show()


