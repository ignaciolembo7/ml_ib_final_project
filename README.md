# ml_ib_final_project

# Resumen del trabajo

![alt text](image.png)



# Carpetas del repositorio. 

- Local: En esta carpeta están los modelos implementados de forma local para luego ser pasados a colab. Aquí es donde realizo modificaciones y pruebas. No están los resultados.

- Colab: Aquí se encuentran los resultados finales, luego de ser compilados por colab. Cada modelo tiene alguna variante en la arquitectura que se describirá a continuación. Los modelos fueron entrenados para los datasets mnist, CIFAR 100 y flowers 102. Al final de cada modelo se realiza una conclusión sobre los resultados obtenidos. 

- EDA: En esta carpeta se realizaron los Exploratory Data Analisys para los datasets mnist y CIFAR 100.

# Definición de los modelos utilizados.

- Modelo 1: 
 
    Tarea:
        cGAN que aprende del mnist y luego genera una imagen fake a partir de un número ingresado por teclado.


    Comentarios:
        Escencialmente la arquitectura es la implementada en https://www.geeksforgeeks.org/conditional-generative-adversarial-network/ donde entrenaron la red para aprender las imagenes del CIFAR10. En nuestro caso, modifiqué la implementación para aprender las imagenes del mnist. Además, agregué diversas medidas para evaluar la performance del modelo. Por último, agregué el código necesario para predecir una imagen de un número ingresado por el usuario.
    probar con regularizadores 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(0.0001)))

Modelo 2: 

    Tarea:
        cGAN que aprende del mnist y luego genera una imagen fake a partir de un número ingresado por teclado.

    Comentarios:
        Uso la arquitectura del paper donde escencialmente hay más capas 
        https://www.tensorflow.org/tutorials/generative/dcgan?hl=es-419

        
        noise = 100 
        encoding = 50 
        lr = 0.0002 
        estandar que usan todos

Conclusiones:
Fue muy positivo cambiar el optimizador de Adam a RMSprop 
No fue demasiado buena la mejora a una red más compleja para el mnist. Hay que ver para el cifar 100 capaz el mnist al ser tan simple la red converge muy rapido. Probar si se ve la mejora en las redes para el cifar 100. Esto se puede argumentar con el pca si se ve que en el EDA el mnist es más fácil de procesar.

Escribir el tiempo que tomo con cada modelo

Optimize todo para np array

En v2 paso a numpy array

el modelo1 es el de geeks, el 2 es el del paper y el 3 es el del paper pero con menos conv transpuestas porque era demasiado tiempo de compilacion, el modelo 4 es igual que el 3 pero en el genereador hay un conv2d al final. En el modelo 4 agrego una capa más al generador con respecto al 3 y cambio el output por una conv en vez con transconv. En 3.1 solo cambio la capa de output del genereador  por una conv en vez con transconv