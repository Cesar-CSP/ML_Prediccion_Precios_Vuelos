# ML_Prediccion_Precios_Vuelos
Proyecto Machine Learning Predicción Precios Vuelos
## **CLA Flight Intelligence**
CLA Flight Intelligence — Travel Agency India es una agencia especializada en vuelos nacionales dentro de India, un mercado donde los precios fluctúan de manera constante debido a factores como la aerolínea, la duración del trayecto, el número de escalas y, especialmente, la antelación con la que se realiza la reserva. Para ofrecer a nuestros clientes las mejores opciones posibles, necesitamos comprender cómo evolucionan los precios y anticipar cuándo es más conveniente comprar un billete.

El objetivo del proyecto es desarrollar un modelo de Machine Learning capaz de predecir el precio esperado de un vuelo nacional en India a partir de sus características principales. Esta predicción permite a CLA Flight Intelligence recomendar a los clientes si es un buen momento para reservar, identificar vuelos con precios inusualmente altos o bajos y optimizar las estrategias comerciales y de marketing de la agencia.

Aunque el dataset cubre únicamente vuelos entre el 11 de febrero y el 31 de marzo, incluye la variable más determinante en el pricing aéreo: la antelación de compra (days_left). Esto permite modelar patrones reales de comportamiento del precio en el corto plazo, suficientes para mejorar la toma de decisiones de la agencia y ofrecer recomendaciones basadas en datos.

El modelo resultante ayuda a CLA Flight Intelligence — Travel Agency India a:

- Recomendar el mejor momento para reservar un vuelo dentro de India,
- Detectar oportunidades de compra cuando un precio está por debajo de lo habitual,
- Priorizar rutas y aerolíneas con mejor relación calidad‑precio,
- Mejorar la experiencia del cliente mediante información personalizada y fundamentada en datos.

Para ello, se ha realizado un preprocesamiento y transformación de las variables, y un pequeño análisis exploratorio de las mismas. Se han creado dos modelos distintos, uno para la clase Economy y otro para la clase Business. Se ha comparado mediante validación cruzada RandomForest (baseline), XGBoost y LightGBM y finalmente se ha escogido el XGBoost como mejor modelo. Se ha procedido a entrenar el modelo con los mejores hiperparámetros dados por una optimización bayesiana y por último, se ha evaluado su rendimiento.

Por otro lado, como científicos de datos de CLA Flight Intelligence creemos que vamos a tener un porblema en los próximos web scrapping porque pensamos que los próximos datos nos vendrán sin la columna de Economy y Bussines, por lo que queremos crear un modelo de clasificación que sea capaz de clasificar los vuelos de economy y Businnes en función de sus otras Features. Todo esto para poder seguir teniendo nuestro dos dataset independientes de economy y business y utilizar el modelo de economy o bussiness que son diferentes. Para ello, se ha seguido un procedimiento análogo al problema de regresión.

### Estructura del repositorio

En el nivel raíz se encuentra el notebook principal (main_Prediccion_Precios_Vuelos_India.ipynb) con el ML, un documento PDF para la exposición del proyecto y una carpeta src con las siguientes subcarpetas:
1. data: contiene los datasets utilizados, economy.csv y business.csv (ambos con las mismas variables), que se han sacado de Kaggle.
2. img: contiene las imágenes utilizadas en la presentación del proyecto.
3. models: modelos en formato joblib.
4. notebooks: contiene los notebooks de desarrollo y en los que se han ido realizando pruebas.
5. utils: contiene un archivo con algunas funciones que han sido utilizadas.

### Librerías y módulos utilizados

1. Python estándar
   - re
   - numpy
   - pandas
2. Visualización
   - matplotlib
   - seaborn
3. Scikit-Learn: Transformadores y Preprocesado
   - BaseEstimator, TransformerMixin
   - ColumnTransformer
   - FunctionTransformer, OrdinalEncoder, OneHotEncoder
   - Pipeline
4. Scikit-Learn: Modelos-Regresión
   - RandomForestRegressor
   - XGBRegressor
   - LGBMRegressor
5. Scikit-Learn: Modelos-Clasificación
   - RandomForestClassifier
   - XGBClassifier
   - LGBMClassifier
6. Scikit-Learn: Métricas Regresión
   - root_mean_squared_error
   - mean_absolute_error
   - r2_score
7. Scikit-Learn: Métricas Clasificación
   - f1_score
   - balanced_accuracy_score
   - roc_auc_score
8. Scikit-Learn: Validación
   - train_test_split
   - cross_validate
   - cross_val_score
   - KFold
9. Scikit-Learn: Selección de features
    - mutual_info_regression
10. Optimización
    - optuna
11. Otros
    - stats
    - joblib
    - os
    - sys

### Instrucciones de reproducción

Lo primero es importar las librerías y cargar los datos, luego cada celda se puede ir ejecutando en orden de arriba a abajo. El notebook main está estructurado en pasos. En el paso 8 se realiza la optimización con optuna, pero optimizar el estudio lleva un rato. Por eso, no es necesario ejecutar esas celdas, simplemente basta con cargar el modelo con `joblib.load` (que ha sido entrenado con los mejores hiperparámetros dados por la optimización) con la ruta correspondiente. Esas celdas solo se han dejado para que se pueda visualizar los pasos seguidos. Lo mismo para el problema de clasificación.

### Principales resultados

**Para el problema de regresión de Economy:**
En primer lugar, la validación cruzada del modelo XGBoost sin optimizar obtuvo un RMSE de 1771 INR ($\approx$ 17€). Tras la optimización, se redujo el RMSE medio de validación cruzada a 1377 INR ($\approx$ 13€), evidenciando una mejora significativa del modelo dentro del conjunto de entrenamiento. Finalmente, el modelo optimizado se entrenó con todo el conjunto de entrenamiento (el 80% original) y se evaluó una única vez sobre el 20% reservado para test, obteniendo un RMSE de 1329 INR ($\approx$ 13€). La proximidad entre el RMSE de validación cruzada (1377) y el RMSE de test (1329) indica que el modelo generaliza correctamente a datos no vistos, sin signos de overfitting ni underfitting. Esto confirma que el rendimiento observado en el conjunto de test es coherente con el comportamiento estimado durante la validación cruzada y valida la calidad del modelo final. Además al comparar los varloes predichos frente a los reales, se observa que el modelo captura adecuadamente la relación entre las variables y el target, que los residuos siguen una distribución normal y que la dispersión va aumentando según aumenta el precio.

**Para el problema de regresión de Business:**
Se obtienen conclusiones similares. XGBoost obtuvo un RMSE de 5536 ($\approx$ 52€) en la validación cruzada sin optimizar. La optimización redujo el RMSE medio de validación cruzada hasta 3931 ($\approx$ 37€), lo que supone una mejora muy significativa respecto al modelo base y confirma que la búsqueda de hiperparámetros permitió capturar mejor la complejidad de los precios en la clase Business. Finalmente, al evaluar contra test se obtiene un RMSE de 3786($\approx$ 36€). Por tanto, la proximidad entre el RMSE de validación cruzada (3931) y el RMSE de test (3786) indica que el modelo generaliza correctamente a datos no vistos, sin signos de overfitting ni underfitting. Esta coherencia entre ambas métricas confirma que el rendimiento observado en el conjunto de test es fiable y que el modelo final es adecuado para predecir precios en la clase Business, a pesar de la mayor variabilidad inherente a esta categoría.

**Para el problema de clasificación:**
En la validación cruzada del modelo XGBoost sin optimizar se obutuvo un score medio F1 de 0.63 aproximadamente. Tras la optimización de hiperparámetros usando la librería Optuna, se mejora a un 0.7. Después de entrenar el modelo, al evaluar el conjunto de test obtenemos un F1 score de 0.71 aproximadamente, por lo tanto el modelo gereraliza bien y no se observa overfitting ni underfitting.

### Autores
- César Sánchez Parra
- Adrián Quindimil Rengel
- Lucía Fuentes González
