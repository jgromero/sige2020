## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2019-2020
## Juan Gómez Romero
## Ejemplo basado en 'Deep Learning with R'
## -------------------------------------------------------------------------------------

library(keras)

## -------------------------------------------------------------------------------------
## Clasificación con red ya entrenada
model_resnet50 <- application_resnet50(
  weights = "imagenet"
)

img_path <- './cat1.jpg'
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)

x <- array_reshape(x, c(1, dim(x)))
x <- imagenet_preprocess_input(x)

preds <- model_resnet50 %>% predict(x)
imagenet_decode_predictions(preds, top = 3)[[1]]

## -------------------------------------------------------------------------------------
## Cargar y pre-procesar imágenes
train_dir      <- './cats_and_dogs_small/train/'
validation_dir <- './cats_and_dogs_small/validation/' 
test_dir       <- './cats_and_dogs_small/test/'

train_datagen      <- image_data_generator(rescale = 1/255) 
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen       <- image_data_generator(rescale = 1/255)

train_data <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary"        # etiquetas binarias
)

validation_data <- flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary"        # etiquetas binarias
)

test_data <- flow_images_from_directory(
  directory = test_dir,
  generator = test_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary"        # etiquetas binarias
)

## -------------------------------------------------------------------------------------
## Extracción de características

# Cargar capa convolutiva de VGG16, pre-entrenada con ImageNet
# https://keras.rstudio.com/reference/application_vgg.html
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# Congelar las capas convolutivas ya entrenada
# https://keras.rstudio.com/reference/freeze_weights.html
freeze_weights(conv_base)

# Acoplar nuevo clasificador (red densa)
model <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# Entrenar modelo (end-to-end)
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 2, # 30,
    validation_data = validation_data,
    validation_steps = 50
  )

plot(history)

# Evaluar modelo
model %>% evaluate_generator(test_data, steps = 50)

## -------------------------------------------------------------------------------------
## Fine tuning (solo con GPU)

# 4. Descongelar una parte de la la capa base
unfreeze_weights(conv_base, from = "block3_conv1")

# 5. Entrenar capa descongelada y FC
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)

history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 25,
    validation_data = validation_data,
    validation_steps = 50
  )

# Evaluar modelo
model %>% evaluate_generator(test_generator, steps = 50)
