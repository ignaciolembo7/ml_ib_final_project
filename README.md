# ml_ib_final_project
 
 Modelo 1: 
 
    Tarea:
        cGAN que aprende del mnist y luego genera una imagen fake a partir de un número ingresado por teclado.


    Comentarios:
        Escencialmente la arquitectura es la implementada en https://www.geeksforgeeks.org/conditional-generative-adversarial-network/ donde entrenaron la red para aprender las imagenes del CIFAR10. En nuestro caso, modifiqué la implementación para aprender las imagenes del mnist. Además, agregué diversas medidas para evaluar la performance del modelo. Por último, agregué el código necesario para predecir una imagen de un número ingresado por el usuario.
    probar con regularizadores 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(0.0001)))