import requests
import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

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
    
    train_df.fillna(np.nan, inplace = True)

    train_df['categoria_de_trabajo'] = train_df['categoria_de_trabajo'].replace(np.nan, 'No respondio')
    train_df['trabajo'] = train_df['trabajo'].replace(np.nan, 'No respondio')
    train_df['barrio'] = train_df['barrio'].replace(np.nan, 'Otro Barrio')

    return train_df

def prepararSetDeValidacion(validation_df:pd.DataFrame):
    #quizas no sea necesaria ya que no se le hace nada a este set
    return validation_df


def oneHotEncodingArbol(df:pd.DataFrame):
    
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    
    
    df = pd.get_dummies(df, drop_first = True, columns = categories)
    
    return df

def ordinalEncodingEducacionAlcanzada(df:pd.DataFrame):
    #no ordena bien segun el anio
    oe = OrdinalEncoder(dtype='int')
    columns_to_encode = ['educacion_alcanzada']
    try:
        df[['educacion_alcanzada_encoded']] = oe.fit_transform(df[columns_to_encode])
    except Exception as upa:
        print(f'Apa lalanga: {upa}')

    return df

def ingenieriaDeFeauturesArboles1(df:pd.DataFrame):
    #dropeamos algunas columnas supongo como las de religion, horas de trabajo registradas
    
    """Hace las transformaciones de datos necesarias para entrenar al arbol de decision."""
    
    
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio'], inplace=True)
    
    df = oneHotEncodingArbol(df)
    df = ordinalEncodingEducacionAlcanzada(df)
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])# se saca la variable target para evitar un leak en el entrenamiento
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder