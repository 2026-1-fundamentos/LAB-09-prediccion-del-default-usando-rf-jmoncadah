# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501

import gzip
import json
import os
import pickle
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def cargar_datasets_comprimidos(ruta_carpeta: str) -> list[pd.DataFrame]:
    """Extrae y carga datasets desde archivos ZIP."""
    lista_dfs = []
    archivos_zip = glob(os.path.join(ruta_carpeta, "*"))

    for archivo in archivos_zip:
        try:
            # Intentar leer directamente como CSV comprimido
            df = pd.read_csv(archivo, compression='zip')
            lista_dfs.append(df)
        except:
            # Fallback: extraer del ZIP
            with zipfile.ZipFile(archivo, "r") as zip_file:
                for contenido in zip_file.namelist():
                    with zip_file.open(contenido) as archivo_csv:
                        df = pd.read_csv(archivo_csv, sep=",")
                        lista_dfs.append(df)

    return lista_dfs


def limpiar_directorio(directorio: str) -> None:
    """Elimina contenido del directorio y lo recrea."""
    if os.path.exists(directorio):
        archivos = glob(os.path.join(directorio, "*"))
        for archivo in archivos:
            try:
                os.remove(archivo)
            except IsADirectoryError:
                pass
        try:
            os.rmdir(directorio)
        except OSError:
            pass

    os.makedirs(directorio, exist_ok=True)


def serializar_modelo_comprimido(ruta_archivo: str, modelo) -> None:
    """Guarda el modelo en formato pickle comprimido con gzip."""
    directorio_padre = os.path.dirname(ruta_archivo)
    limpiar_directorio(directorio_padre)

    with gzip.open(ruta_archivo, "wb") as archivo_gz:
        pickle.dump(modelo, archivo_gz)


def limpiar_datos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Limpia y transforma el dataset según las especificaciones."""
    datos = dataframe.copy()

    # Renombrar columna objetivo
    datos = datos.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    if 'ID' in datos.columns:
        datos = datos.drop(columns=['ID'])

    # Eliminar registros con información no disponible
    datos = datos.dropna()

    # Filtrar EDUCATION y MARRIAGE != 0
    datos = datos.loc[datos["MARRIAGE"] != 0]
    datos = datos.loc[datos["EDUCATION"] != 0]

    # Agrupar niveles superiores de educación
    datos["EDUCATION"] = datos["EDUCATION"].apply(
        lambda valor: 4 if valor > 4 else valor
    )

    return datos


def dividir_caracteristicas_objetivo(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el dataset en características (X) y variable objetivo (y)."""
    caracteristicas = dataframe.drop(columns=["default"])
    objetivo = dataframe["default"]
    return caracteristicas, objetivo


def construir_pipeline_optimizacion() -> GridSearchCV:
    """Crea pipeline con OneHotEncoder y RandomForest, y configura GridSearch."""
    # Variables categóricas para codificación
    variables_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    
    # Variables numéricas para escalado
    variables_numericas = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    # Transformador de columnas con OneHotEncoder y MinMaxScaler
    transformador = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), variables_categoricas),
            ("numericas", MinMaxScaler(), variables_numericas)
        ]
    )

    # Clasificador
    bosque_aleatorio = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Pipeline completo
    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", transformador),
            ("clasificador", bosque_aleatorio),
        ]
    )

    # Grilla de hiperparámetros optimizada
    parametros = {
        "clasificador__n_estimators": [100, 200],
        "clasificador__max_depth": [10, 15, None],
        "clasificador__min_samples_split": [5, 10],
        "clasificador__min_samples_leaf": [2, 4],
        "clasificador__class_weight": ['balanced', None]
    }

    # Búsqueda con validación cruzada
    optimizador = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=10,  # 10 splits como se solicita
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    return optimizador


def calcular_metricas_rendimiento(nombre_conjunto: str, valores_reales, valores_predichos) -> dict:
    """Calcula métricas de evaluación del modelo."""
    metricas = {
        "type": "metrics",
        "dataset": nombre_conjunto,
        "precision": float(precision_score(valores_reales, valores_predichos, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(valores_reales, valores_predichos)),
        "recall": float(recall_score(valores_reales, valores_predichos, zero_division=0)),
        "f1_score": float(f1_score(valores_reales, valores_predichos, zero_division=0)),
    }
    return metricas


def generar_matriz_confusion(nombre_conjunto: str, valores_reales, valores_predichos) -> dict:
    """Genera diccionario con la matriz de confusión."""
    matriz = confusion_matrix(valores_reales, valores_predichos)

    resultado = {
        "type": "cm_matrix",
        "dataset": nombre_conjunto,
        "true_0": {
            "predicted_0": int(matriz[0][0]),
            "predicted_1": int(matriz[0][1])
        },
        "true_1": {
            "predicted_0": int(matriz[1][0]),
            "predicted_1": int(matriz[1][1])
        },
    }

    return resultado


def ejecutar_pipeline_completo() -> None:
    """Función principal que ejecuta todo el flujo de trabajo."""
    try:
        print("Cargando datasets...")
        # Cargar y limpiar datos
        datasets_crudos = cargar_datasets_comprimidos("files/input")
        
        if len(datasets_crudos) < 2:
            # Fallback: cargar directamente si no hay archivos ZIP
            try:
                train_df = pd.read_csv("files/input/train_data.csv.zip", compression='zip')
                test_df = pd.read_csv("files/input/test_data.csv.zip", compression='zip')
                datasets_crudos = [train_df, test_df]
            except:
                train_df = pd.read_csv("train_data.csv.zip", compression='zip')
                test_df = pd.read_csv("test_data.csv.zip", compression='zip')
                datasets_crudos = [train_df, test_df]
        
        print("Limpiando datos...")
        datasets_limpios = [limpiar_datos(dataset) for dataset in datasets_crudos]

        # Separar en conjuntos de prueba y entrenamiento
        datos_entrenamiento, datos_prueba = datasets_limpios

        print(f"Train shape: {datos_entrenamiento.shape}")
        print(f"Test shape: {datos_prueba.shape}")

        # Dividir características y objetivo
        X_entrenamiento, y_entrenamiento = dividir_caracteristicas_objetivo(datos_entrenamiento)
        X_prueba, y_prueba = dividir_caracteristicas_objetivo(datos_prueba)

        print("Entrenando modelo con optimización...")
        # Entrenar modelo con optimización de hiperparámetros
        modelo_optimizado = construir_pipeline_optimizacion()
        modelo_optimizado.fit(X_entrenamiento, y_entrenamiento)

        print(f"Mejores parámetros: {modelo_optimizado.best_params_}")
        print(f"Mejor score: {modelo_optimizado.best_score_:.4f}")

        # Guardar modelo
        ruta_modelo = os.path.join("files", "models", "model.pkl.gz")
        serializar_modelo_comprimido(ruta_modelo, modelo_optimizado)
        print(f"Modelo guardado en: {ruta_modelo}")

        # Generar predicciones
        print("Generando predicciones...")
        predicciones_prueba = modelo_optimizado.predict(X_prueba)
        predicciones_entrenamiento = modelo_optimizado.predict(X_entrenamiento)

        # Calcular métricas
        print("Calculando métricas...")
        metricas_entrenamiento = calcular_metricas_rendimiento(
            "train", y_entrenamiento, predicciones_entrenamiento
        )
        metricas_prueba = calcular_metricas_rendimiento(
            "test", y_prueba, predicciones_prueba
        )

        # Calcular matrices de confusión
        confusion_entrenamiento = generar_matriz_confusion(
            "train", y_entrenamiento, predicciones_entrenamiento
        )
        confusion_prueba = generar_matriz_confusion(
            "test", y_prueba, predicciones_prueba
        )

        # Guardar resultados
        Path("files/output").mkdir(parents=True, exist_ok=True)

        ruta_metricas = "files/output/metrics.json"
        with open(ruta_metricas, "w", encoding="utf-8") as archivo_metricas:
            archivo_metricas.write(json.dumps(metricas_entrenamiento) + "\n")
            archivo_metricas.write(json.dumps(metricas_prueba) + "\n")
            archivo_metricas.write(json.dumps(confusion_entrenamiento) + "\n")
            archivo_metricas.write(json.dumps(confusion_prueba) + "\n")

        print(f"Resultados guardados en: {ruta_metricas}")

        # Mostrar métricas finales
        print("\n=== MÉTRICAS FINALES ===")
        print(f"Train - Precision: {metricas_entrenamiento['precision']:.4f}")
        print(f"Test  - Precision: {metricas_prueba['precision']:.4f}")
        print(f"Train - Balanced Accuracy: {metricas_entrenamiento['balanced_accuracy']:.4f}")
        print(f"Test  - Balanced Accuracy: {metricas_prueba['balanced_accuracy']:.4f}")

        print("¡Proceso completado exitosamente!")

    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ejecutar_pipeline_completo()