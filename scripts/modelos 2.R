cv <- trainControl(method = "cv", number = 10)

# Entrenar el modelo glmboost
modelo_glmboost <- train(
  formula = price ~ bedrooms + bathrooms + rooms + ascensor + patio + remodel + parqueo + piso +
    as.factor(year) + as.factor(property_type) + iluminado + distancia_parque +
    distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco +
    distancia_bus + ESTRATO,
  data = train_final,
  method = "glmboost",
  trControl = cv
)

# Visualizar el resumen del modelo
summary(modelo_glmboost)

 #entrenar modelo de red neuronal 
modelo_nnet <- train(
  formula = price ~ bedrooms + bathrooms + rooms + ascensor + patio + remodel + parqueo + piso +
    as.factor(year) + as.factor(property_type) + iluminado + distancia_parque +
    distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco +
    distancia_bus + ESTRATO,
  data = train_final,
  method = "nnet",
  trControl = cv,
  linout = TRUE,     # Para regresión en lugar de clasificación
  trace = FALSE,     # Para no mostrar el proceso de entrenamiento
  tuneLength = 5     # Para buscar los mejores hiperparámetros
)
 

# Definir el grid de hiperparámetros
tune_grid <- expand.grid(
  size = c(5, 10, 15),  # Número de neuronas en la capa oculta
  decay = c(0.001, 0.01, 0.1)  # Tasa de decaimiento
)

# Entrenar el modelo de red neuronal
modelo_nnet <- train(
  formula = price ~ bedrooms + bathrooms + rooms + ascensor + patio + remodel + parqueo + piso +
    as.factor(year) + as.factor(property_type) + iluminado + distancia_parque +
    distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco +
    distancia_bus + ESTRATO,
  data = train_final,
  method = "nnet",
  trControl = cv_control,
  tuneGrid = tune_grid,
  linout = TRUE,     # Para regresión en lugar de clasificación
  trace = FALSE      # Para no mostrar el proceso de entrenamiento
)

# Resumen del modelo
print(modelo_nnet)

