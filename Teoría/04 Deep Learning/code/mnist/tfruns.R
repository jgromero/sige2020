## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2019-2020
## Juan Gómez Romero
## -------------------------------------------------------------------------------------

if( ! ("tfruns" %in%  installed.packages()[,"Package"]) ) {
  install.packages("tfruns")  
}

library(tfruns)

# Lanzar script de entrenamiento
training_run("mnist-cnn_tfruns.R")

# Consultar resultados (modo texto)
latest_run()

# Visualizar ejecución en navegador
view_run()

## Modificar fichero "mnist-cnn_tfruns.R" para usar un modelo mejorado 
## con Batch Normalization y regularización
##
# model %>% 
#   layer_conv_2d(filters = 20, kernel_size = c(5, 5), activation = "relu", input_shape = c(28, 28, 1)) %>%
#   layer_batch_normalization(epsilon = 0.01) %>%
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#   layer_flatten() %>%
#   layer_dense(units = 100, activation = "sigmoid", kernel_regularizer = regularizer_l2(0.01)) %>%
#   layer_dense(units = 10, activation = "softmax")

# Lanzar de nuevo script de entrenamiento
training_run("mnist-cnn_tfruns.R")

# Comparar las dos últimas ejecuciones
compare_runs() 

## Cambiar fichero "mnist-cnn_tfruns.R" para dropout en lugar de regularización
##
# model %>% 
#   layer_conv_2d(filters = 20, kernel_size = c(5, 5), activation = "relu", input_shape = c(28, 28, 1)) %>%
#   layer_batch_normalization(epsilon = 0.01) %>%
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#   layer_flatten() %>%
#   layer_dense(units = 100, activation = "sigmoid") %>%
#   layer_dropout(rate = 0.4) %>%
#   layer_dense(units = 10, activation = "softmax")

# Lanzar de nuevo script de entrenamiento
training_run("mnist-cnn_tfruns.R")

# Comparar las dos últimas ejecuciones
compare_runs() 

# Listar resultados de un directorio
ls_runs(order = eval_accuracy)

# Lanzar script parametrizado 
training_run("mnist-cnn_tfruns-params.R")

# Lanzar script parametrizado con nuevos parámetros
training_run("mnist-cnn_tfruns-params.R", 
             flags = list(dropout=0.5, hidden_activation ="relu", batch_size=64))

# Comparar las dos últimas ejecuciones
compare_runs() 

# Lanzar conjunto de experimentos parametrizados
# Para ejecutar desde terminal: Rscript -e 'tfruns::tuning_run(<parameters>)'
runs <- tuning_run("mnist-cnn_tfruns-params.R", 
             runs_dir = "dropout_batch_tuning",
             flags = list(dropout=c(0.2, 0.3, 0.4, 0.5),
                          hidden_activation ="relu",
                          batch_size=c(64, 128, 256)))   # solo una muestra: sample = 0.3

# Listar por orden de resultado
runs_sorted <- runs[order(runs$eval_accuracy, decreasing = TRUE), ]

# Comparar los dos mejores
compare_runs(c(runs_sorted[1,]$run_dir, runs_sorted[2,]$run_dir))