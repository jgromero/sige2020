## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2019-2020
## Juan Gómez Romero
## -------------------------------------------------------------------------------------

library(keras)
library(mlflow)

hidden_units      <- mlflow_param("hidden_units", 100, "integer", "Number of units of the hidden layer")
hidden_activation <- mlflow_param("hidden_activation", "sigmoid", "string", "Activation function for the hidden layer")
dropout           <- mlflow_param("dropout", 0.3, "numeric", "Dropout rate (after the hidden layer)")
epsilon           <- mlflow_param("epsilon", 0.01, "numeric", "Epsilon parameter of the batch normalization (after convolution)")
batch_size        <- mlflow_param("batch_size", 128, "integer", "Mini-batch size")
epochs            <- mlflow_param("epochs", 5, "integer", "Number of training epochs")

## -------------------------------------------------------------------------------------
## Cargar y pre-procesar datos

# Cargar MNIST
mnist <- dataset_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y

# Redimensionar imagenes
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))  # 60.000 matrices 28x28x1
x_test  <- array_reshape(x_test,  c(nrow(x_test),  28, 28, 1))  # 60.000 matrices 28x28x1

# Reescalar valores de imagenes a [0, 255]
x_train <- x_train / 255
x_test  <- x_test  / 255

# Crear 'one-hot' encoding
y_train <- to_categorical(y_train, 10)
y_test  <- to_categorical(y_test,  10)

## -------------------------------------------------------------------------------------
## Crear modelo

# Definir arquitectura
model <- keras_model_sequential() 
model %>% 
 layer_conv_2d(filters = 20, kernel_size = c(5, 5), activation = "relu", input_shape = c(28, 28, 1)) %>%
 layer_batch_normalization(epsilon = epsilon) %>%
 layer_max_pooling_2d(pool_size = c(2, 2)) %>%
 layer_flatten() %>%
 layer_dense(units = hidden_units, activation = hidden_activation) %>%
 layer_dropout(rate = dropout) %>%
 layer_dense(units = 10, activation = "softmax")

summary(model)

# Compilar modelo
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

## -------------------------------------------------------------------------------------
## MLflow
with(mlflow_start_run(), {

  # Entrenar modelo
  history <- model %>% 
    fit(
      x_train, y_train, 
      epochs = epochs, 
      batch_size = batch_size,
      validation_split = 0.2
    )
  
  # Visualizar entrenamiento
  plot(history)
  
  # Calcular metricas sobre datos de validación
  metrics <- model %>% 
    evaluate(x_test, y_test)
  
  # Guardar valores interesantes de la ejecución
  # Por ejemplo, para estudio de dropout + epochs
  mlflow_log_param("dropout", dropout)
  mlflow_log_param("epochs", epochs)
  mlflow_log_metric("loss", metrics$loss)
  mlflow_log_metric("accuracy", metrics$accuracy)
  
  # Guardar modelo
  mlflow_log_model(model, "model")
  
  # Mostrar salida
  message("CNN model (dropout=", dropout, ", epochs=", epochs, "):")
  message("  loss: ", metrics$loss)
  message("  accuracy: ", metrics$accuracy)
})