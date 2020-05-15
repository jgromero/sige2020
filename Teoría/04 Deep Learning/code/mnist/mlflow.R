## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2019-2020
## Juan Gómez Romero
## -------------------------------------------------------------------------------------

if( ! ("mlflow" %in%  installed.packages()[,"Package"]) ) {
  install.packages("mlflow")
  library(mlflow)
  mlflow::install_mlflow()
}

library(mlflow)

# Lanzar script de entrenamiento
mlflow_run(entry_point = "mnist-cnn_mlflow.R")

# Visualizar en interfaz MLflow
# http://127.0.0.1:5987/
mlflow_ui()

# Desplegar el modelo disponible
# http://127.0.0.1:8090/
mlflow_rfunc_serve(model_uri="mlruns/0/0228e8afb42f48e5bff92defc4a5b28a/artifacts/model", port=8090)

# Cambiar valores de los parámetros
mlflow_run(entry_point = "mnist-cnn_mlflow.R", parameters = list(dropout = 0.5, epochs = 3))
