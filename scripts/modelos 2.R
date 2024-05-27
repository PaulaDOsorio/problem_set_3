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

