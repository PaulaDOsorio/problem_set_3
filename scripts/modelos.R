setwd("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts")

#rm(list = ls())

# - Librerias y paquetes 

library(pacman)
p_load(tidyverse, # Manipular dataframes
       rstudioapi, 
       rio, # Import data easily
       leaflet, # Mapas interactivos
       rgeos, 
       tmaptools, # geocode_OSM()
       sf, # Leer/escribir/manipular datos espaciales
       stargazer,
       osmdata, # Get OSM's data 
       ranger,
       gmb,
       glmnet, # To implement regularization algorithms. 
       caret, # creating predictive models
       plotly) # Gráficos interactivos

#train_final<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/train_final.csv")
#test_final<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/test_final.csv")

## Probando con la variable estrato ##
train_final<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/train_final_estrato_sin_ms.csv")
test_final<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/test_final_estrato_sin_ms.csv")

#train_final$remodel <- as.factor(train_final$remodel)
#train_final$ascensor <-  as.factor(train_final$ascensor)
#train_final$iluminado <- as.factor(train_final$iluminado)
#train_final$parqueo <- as.factor(train_final$parqueo) 
#train_final$patio <- as.factor(train_final$patio) 
#train_final$deposito <- as.factor(train_final$deposito) 

#test_final$remodel <- as.factor(test_final$remodel)
#test_final$ascensor <-  as.factor(test_final$ascensor)
#test_final$iluminado <- as.factor(test_final$iluminado)
#test_final$parqueo <- as.factor(test_final$parqueo) 
#test_final$patio <- as.factor(test_final$patio) 
#test_final$deposito <- as.factor(test_final$deposito) 

# Ver cuantos datos faltantes hay en la variable de superficie y price
missing1 <- is.na(train_final$surface_total)
sum(missing1)

missing2 <- is.na(train_final$price)
sum(missing2)


# Partimos la base de Train final en 2: t_train y t_test 

set.seed(1234)

inTrain <- createDataPartition(
  y = train_final$price,## La variable dependiente u objetivo 
  p = .7, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)


t_train <- train_final[ inTrain,]
t_test  <- train_final[-inTrain,]

# Eslacar los datos ----
variables_numericas <- c("distancia_parque", "distancia_escuela", 
                         "distancia_estacion","distancia_comercial", 
                         "distancia_banco", "distancia_bus")

escalador1 <- preProcess(train_final[, variables_numericas],
                         method = c("center", "scale"))

train_final[, variables_numericas] <- predict(escalador1, train_final[, variables_numericas])
test_final[, variables_numericas] <- predict(escalador1, test_final[, variables_numericas])


escalador2 <- preProcess(t_train[, variables_numericas],
                         method = c("center", "scale"))

t_train[, variables_numericas] <- predict(escalador1, t_train[, variables_numericas])
t_test[, variables_numericas] <- predict(escalador1, t_test[, variables_numericas])


filtro3 <- is.na(t_train$surface_total)
sum(filtro3)
filtro4 <- is.na(t_train$price)
sum(filtro4)
# Variables con missing todavía
sapply(t_train, function(x) sum(is.na(x)))
sapply(t_test, function(x) sum(is.na(x)))

# Tienen missing las siguientes variables:
#title:12
#description:5
# description_num: 5
# remodel: 5
# ascensor: 5
# iluminado: 5
# parqueo:5  
# patio: 5          
# deposito:5
## Modelos ##

# Selecciona solo las columnas que mencionaste con valores faltantes
columnas_con_missing <- c("title", "description", "description_num", "remodel", "ascensor", "iluminado", "parqueo", "patio", "deposito")

# Elimina las filas que tienen cualquier valor faltante en esas columnas
t_train <- t_train[!rowSums(is.na(t_train[, columnas_con_missing])), ]

# Elimina las filas que tienen cualquier valor faltante en esas columnas
t_test <- t_test[!rowSums(is.na(t_test[, columnas_con_missing])), ]


## Valores faltantes en la test original ##
sapply(test_final, function(x) sum(is.na(x)))

# Tienen missing las siguientes variables:
#title:6
#description:2
# description_num: 2
# remodel: 2
# ascensor: 2
# iluminado: 2
# parqueo:2  
# patio: 2          
# deposito:2
## Modelos ##

# Selecciona solo las columnas que mencionaste con valores faltantes

library(dplyr)

vars_texto <- c("title", "description", "description_num")
vars_numer_categ <- c("remodel", "ascensor", "iluminado", "parqueo", "patio", "deposito")

test_final <- test_final %>%
  mutate(
    across(all_of(vars_texto), ~replace_na(., "")),  # Reemplazar NA por cadenas vacías en variables de texto
    across(all_of(vars_numer_categ), ~replace_na(., 0))  # Reemplazar NA por cero en las numéricas/categóricas
  )



p_load(MLmetrics)

## Modelo 1: Regresión lineal ##
cv1 <- trainControl(number = 5, method = "cv")

modelo_lm <- train(price ~ bedrooms + parqueo + as.factor(year) + patio  + distancia_parque + distancia_escuela 
                   + distancia_estacion + distancia_comercial + as.factor(ESTRATO), 
                   data = t_train, 
                   method = "lm",
                   trControl = cv1
)
modelo_lm

## Métricas modelo lineal - 1
y_hat_outsample1 = predict(modelo_lm, newdata = t_test)
MAE(y_pred = y_hat_outsample1, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample1, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample1, y_true = t_test$price)

## Modelo 2: Arbol de decision ##
cv <- trainControl(number = 10, method = "cv")

# especificamos la grilla de los alphas
grid <- expand.grid(cp = seq(0, 0.03, 0.001))

modelo_arbol_decision <- train(price ~ rooms + bedrooms + parqueo + as.factor(year) + as.factor(property_type) 
                               + patio + remodel + iluminado + distancia_parque 
                               + distancia_escuela + distancia_estacion + 
                                 distancia_comercial + distancia_banco + ESTRATO,
                               data = t_train, 
                               method = "rpart", 
                               trControl = cv,
                               tuneGrid = grid)
modelo_arbol_decision

## Métricas modelo:Arbol de decision
y_hat_outsample2 = predict(modelo_arbol_decision, newdata = t_test)
MAE(y_pred = y_hat_outsample2, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample2, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample2, y_true = t_test$price)

# Preparando envio a Kaggle del modelo:Arbol de decision

predictSample_rf <- test_final   %>% 
  mutate(price = predict(modelo_arbol_decision, newdata = test_final, type = "raw")    ## predicted precio de la vivienda 
  )  %>% select(property_id, price)

head(predictSample_rf)

#Es consistente con el template
template<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/submission_template.csv")

head(template)
setwd("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/")

#predictSample_rf <- predictSample_rf %>% select(property_id, price)
write.csv(predictSample_rf,"stores/Prediction_Arbol_Decision.csv", row.names = FALSE)

## Modelo 3: GBM ##
#grid_gbm <- expand.grid(n.trees = c(500, 1000),
#                        interaction.depth = c(4, 5, 6),
#                        shrinkage = c(0.01, 0.1),
#                        n.minobsinnode = c(10, 20))

grid_gbm<-expand.grid(n.trees=1000,interaction.depth=5, shrinkage=0.01, n.minobsinnode = 20)
modelo_GBM <- train(price ~ rooms + bedrooms + patio + remodel + iluminado + distancia_parque 
                    + distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco
                    + distancia_bus + ESTRATO,
                    data = t_train, 
                    method = "gbm", 
                    trControl = cv1,
                    metric = "MAE",
                    tuneGrid = grid_gbm
)

modelo_GBM 

## Métricas modelo GBN
y_hat_outsample3 = predict(modelo_GBM, newdata = t_test)
MAE(y_pred = y_hat_outsample3, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample3, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample3, y_true = t_test$price)

## Modelo 4: Ramdon Forest ##
cv <- trainControl(method = "cv", number = 10, search = "grid")
tunegrid_rf <- expand.grid(mtry = 5, 
                           min.node.size = 10,
                           splitrule = "variance")

modelo_Ramdon_Forest <- train(price ~ bedrooms + bathrooms + rooms + ascensor + patio + remodel 
                              + as.factor(year) + as.factor(property_type) + iluminado + distancia_parque 
                              + distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco
                              + distancia_bus + ESTRATO, 
                              data = t_train,
                              method = "ranger", 
                              trControl = cv,
                              metric = 'MAE', 
                              tuneGrid = tunegrid_rf)

modelo_Ramdon_Forest

## Métricas modelo Ramdon Forest
y_hat_outsample4 = predict(modelo_Ramdon_Forest, newdata = t_test)
MAE(y_pred = y_hat_outsample4, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample4, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample4, y_true = t_test$price)

## Preparando envio a Kaggle modelo Ramdon Forest ##

predictSample_rf1 <- test_final   %>% 
  mutate(price = predict(modelo_Ramdon_Forest, newdata = test_final, type = "raw")    ## predicted precio de la vivienda 
  )  %>% select(property_id, price)

head(predictSample_rf1)

#Es consistente con el template
template<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/submission_template.csv")

head(template)
setwd("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/")

#predictSample_rf <- predictSample_rf %>% select(property_id, price)
write.csv(predictSample_rf1,"stores/Prediction_Ramdon_Forest_F.csv", row.names = FALSE) ## ajustar predictSample_rf


## Modelo 5: Ramdon Forest con mas variables de control ##
cv <- trainControl(method = "cv", number = 10, search = "grid")
modelo_RF2 <- train(price ~ bedrooms + bathrooms + rooms + ascensor + patio + remodel + parqueo + piso
                    + as.factor(property_type) + iluminado + distancia_parque 
                    + distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco
                    + distancia_bus + ESTRATO,
                    data = t_train, 
                    method = "ranger", 
                    trControl = cv,
                    metric = 'MAE', 
                    tuneGrid = tunegrid_rf)

modelo_RF2

## Métricas Ramdon Forest con más varables de control
y_hat_outsample5 = predict(modelo_RF2, newdata = t_test)
MAE(y_pred = y_hat_outsample5, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample5, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample5, y_true = t_test$price)

#Preparando envio a Kaggle modelo Ramdon Forest con más controles

predictSample_rf2 <- test_final   %>% 
  mutate(price = predict(modelo_RF2, newdata = test_final, type = "raw")    ## predicted precio de la vivienda 
  )  %>% select(property_id, price)

head(predictSample_rf2)

#Es consistente con el template
template<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/submission_template.csv")

head(template)
setwd("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/")

#predictSample_rf <- predictSample_rf %>% select(property_id, price)
#write.csv(predictSample_rf2,"stores/Prediction_RF_+controles.csv", row.names = FALSE)
write.csv(predictSample_rf2,"stores/Prediction_RF_+controles_F.csv", row.names = FALSE)

## Modelo 6: Regresión Lasso ##
cv2 <- trainControl(method = "cv", number = 10, search = "grid")

lambda_grid <- 10^seq(-4, 0.01, length = 200) 
modelo_Lasso <- train(price ~ bedrooms + bathrooms + rooms + ascensor + patio + parqueo + remodel + piso
                      + as.factor(property_type) + iluminado + distancia_parque 
                      + distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco
                      + as.factor(year) + distancia_bus + ESTRATO, 
                      data = t_train, 
                      method = "glmnet",
                      trControl = cv2,
                      metric = "MAE",
                      tuneGrid = expand.grid(alpha = 1,lambda=lambda_grid), 
                      preProcess = c("center", "scale")
)

modelo_Lasso
print(modelo_Lasso)

## Métricas modelo Regresión Lasso
y_hat_outsample6 = predict(modelo_Lasso, newdata = t_test)
MAE(y_pred = y_hat_outsample6, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample6, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample6, y_true = t_test$price)

#Preparando envio a Kaggle modelo Regresión Lasso

predictSample_rf4 <- test_final   %>% 
  mutate(price = predict(modelo_Lasso, newdata = test_final, type = "raw")    ## predicted precio de la vivienda 
  )  %>% select(property_id, price)

head(predictSample_rf4)

#Es consistente con el template
template<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/submission_template.csv")

head(template)
setwd("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/")

#predictSample_rf <- predictSample_rf %>% select(property_id, price)
write.csv(predictSample_rf4,"stores/Prediction_Lasso.csv", row.names = FALSE)


## Modelo 7: Regresión Ridge ##
cv2 <- trainControl(method = "cv", number = 10, search = "grid")
lambda_grid <- 10^seq(-4, 0.01, length = 200)
modelo_Ridge <- train(price ~ bedrooms + bathrooms + rooms + ascensor + patio + remodel + parqueo + piso
                      + as.factor(property_type) + iluminado + distancia_parque 
                      + distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco
                      + as.factor(year) + distancia_bus + ESTRATO, 
                      data = t_train, 
                      method = "glmnet",
                      trControl = cv2,
                      metric = "MAE",
                      tuneGrid = expand.grid(alpha = 0,lambda=lambda_grid), 
                      preProcess = c("center", "scale")
)

modelo_Ridge 

## Métricas modelo Regresión Ridge
y_hat_outsample7 = predict(modelo_Ridge, newdata = t_test)
MAE(y_pred = y_hat_outsample7, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample7, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample7, y_true = t_test$price)

######
install.packages("doParallel")
library(doParallel)
registerDoParallel(cores = detectCores())  # Registra todos los núcleos disponibles
#cv2 <- trainControl(method = "cv", number = 10, search = "grid", allowParallel = TRUE)

## Modelo 8: Random Forest & Expansion grid ##
cv2 <- trainControl(method = "cv", number = 10,search = "grid")
tunegrid_rf2 <- expand.grid(mtry = 8)

modelo_RF_Grid <- train(price ~ bedrooms + bathrooms + rooms + ascensor + patio + remodel + parqueo + piso
                        + as.factor(property_type) + iluminado + distancia_parque 
                        + distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco
                        + as.factor(year) + distancia_bus + ESTRATO,
                        data = t_train,
                        method = "rf", 
                        trControl = cv2,
                        tuneGrid = tunegrid_rf2,
                        metric = 'MAE',
                        ntree = 200
)

modelo_RF_Grid

## Métricas modelo Random Forest & Expansion grid
y_hat_outsample8 = predict(modelo_RF_Grid, newdata = t_test)
MAE(y_pred = y_hat_outsample8, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample8, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample8, y_true = t_test$price)


## Preparando envio a Kaggle modelo Random Forest & Expansion grid##

predictSample_rf3 <- test_final   %>% 
  mutate(price = predict(modelo_RF_Grid, newdata = test_final, type = "raw")    ## predicted precio de la vivienda 
  )  %>% select(property_id, price)

head(predictSample_rf3)

#Es consistente con el template
template<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/submission_template.csv")

head(template)
setwd("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/")

#predictSample_rf <- predictSample_rf %>% select(property_id, price)
#write.csv(predictSample_rf3,"stores/Prediction_RF_Grid.csv", row.names = FALSE)
write.csv(predictSample_rf3,"stores/Prediction_RF_Grid_F.csv", row.names = FALSE)
#write.csv(predictSample_rf3,"stores/Prediction_RF_Grid_VF.csv", row.names = FALSE)


## Modelo 9: Elastic Net ##
modelo_Elastic_Net <-train(price ~ bedrooms + bathrooms + rooms + ascensor + patio + remodel + parqueo + piso
                           + iluminado + ascensor + distancia_parque 
                           + distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco
                           + as.factor(year) + distancia_bus + ESTRATO,
                           data=t_train,
                           method = 'glmnet', 
                           trControl = cv2,
                           tuneGrid = expand.grid(alpha = seq(0,1, by = 0.1), 
                                                  lambda = seq(0.001,0.02,by = 0.001)),
                           preProcess = c("center", "scale")
) 

modelo_Elastic_Net

## Métricas modelo Elastic Net 
y_hat_outsample9 <- predict(modelo_Elastic_Net, newdata = t_test)
MAE(y_pred = y_hat_outsample9, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample9, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample9, y_true = t_test$price)

## Modelo 10: Bagging ##

p_load(ipred)
modelo_Bagging <- bagging(price ~ bedrooms + bathrooms + rooms + ascensor + patio + remodel + parqueo + piso
                          + iluminado + ascensor + distancia_parque 
                          + distancia_escuela + distancia_estacion + distancia_comercial + distancia_banco
                          + distancia_bus + ESTRATO,
                          data  = t_train, nbagg = 500)

modelo_Bagging

## Métricas modelo Bagging
y_hat_outsample10 <- predict(modelo_Bagging, newdata = t_test)
MAE(y_pred = y_hat_outsample10, y_true = t_test$price)
MAPE(y_pred = y_hat_outsample10, y_true = t_test$price)
RMSE(y_pred = y_hat_outsample10, y_true = t_test$price)


## Preparando envio a Kaggle modelo Bagging ##

predictSample_rf5 <- test_final   %>% 
  mutate(price = predict(modelo_Bagging, newdata = test_final, type = "raw")    ## predicted precio de la vivienda 
  )  %>% select(property_id, price)

head(predictSample_rf5)

#Es consistente con el template
template<-read.csv("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/scripts/submission_template.csv")

head(template)
setwd("C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/")

#predictSample_rf <- predictSample_rf %>% select(property_id, price)
#write.csv(predictSample_rf5,"stores/Prediction_Bagging.csv", row.names = FALSE)
write.csv(predictSample_rf5,"stores/Prediction_Bagging_F.csv", row.names = FALSE)

## Luego de correr varios modelos, el mejor modelo resultante es el modelo 7: Random Forest & Expansion grid 
# es el que resultó con el MAE más bajo de 114992066
mejor_modelo = modelo_RF_Grid
y_predict <- predict(mejor_modelo, newdata = test_final)

mejormodelofinal <- data.frame(
  property_id = test_final$property_id,
  price = y_predict     
)
mejormodelofinal
write.csv(mejormodelofinal, "C:/Users/sandr/Documents/GitHub/BIG DATA/Taller3/stores/mejormodelofinal.csv", row.names =  FALSE)

################

