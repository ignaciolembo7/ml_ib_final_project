# ml_ib_final_project
 
Modelo 1: 
 
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