import requests
import numpy as np
import pandas as pd

DATA_VALIDATION = "impuestos_val.csv"
DATA_TRAIN = "impuestos_train.csv"
DATA_VALIDATION_URL = "https://docs.google.com/spreadsheets/d/1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE/export?format=csv"
DATA_TRAIN_URL = "https://docs.google.com/spreadsheets/d/1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0/export?format=csv"

def cargarDatasets():

    with requests.get(
    DATA_TRAIN_URL
    ) as r, open(DATA_TRAIN, "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)

    x = pd.read_csv(DATA_TRAIN)

    with requests.get(
    DATA_VALIDATION_URL
    ) as r, open(DATA_VALIDATION, "wb") as f:
        for chunk in r.iter_content():
            f.write(chunk)

    y = pd.read_csv(DATA_VALIDATION)
    return x,y

def prepararSetDeEntrenamiento(train_df:pd.DataFrame):
    #dropeamos algunas columnas supongo como las de religion, horas de trabajo registradas
    #no se si estaria bien borrar las filas que borrabamos en el primer tp
    train_df.drop(columns=['religion','horas_trabajo_registradas'], inplace=True)
    train_df.fillna(np.nan, inplace = True)

    train_df['categoria_de_trabajo'] = train_df['categoria_de_trabajo'].replace(np.nan, 'No respondio')
    train_df['trabajo'] = train_df['trabajo'].replace(np.nan, 'No respondio')
    train_df['barrio'] = train_df['barrio'].replace(np.nan, 'Otro Barrio')

    #train_df = conversionDeVariablesCategoricas(train_df)

    return train_df

def prepararSetDeValidacion(validation_df:pd.DataFrame):
    #borro el id porque no se q es seguro no sirve
    #representatividad_poblacional  tampoco se q es wtf
    validation_df.drop(columns=['religion','id','representatividad_poblacional'], inplace=True)
    return validation_df
