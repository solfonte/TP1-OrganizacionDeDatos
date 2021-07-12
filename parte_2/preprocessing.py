import requests
import numpy as np
import pandas as pd
from sklearn import preprocessing

from funcionesAuxiliares import *


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


def ingenieriaDeFeaturesArboles1(df:pd.DataFrame):    
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
              'rol_familiar_registrado', 'trabajo']
    
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    
    df.drop(columns=
      ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','anios_estudiados'],inplace=True)
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder


def ingenieriaDeFeaturesArboles2(df:pd.DataFrame):
    
    df = reducirTrabajos(df)
    df = reducirCategorias(df)
    df = reducirEstadoMarital(df)
    
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','anios_estudiados'], inplace=True)
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder 

def ingenieriaDeFeaturesKnn(df:pd.DataFrame):
    
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada'], inplace=True)
    
    df = normalizar(df)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder

def ingenieriaDeFeaturesSVM(df:pd.DataFrame):
    
    categories = ['estado_marital', 'genero', 'trabajo']
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','rol_familiar_registrado', 'categoria_de_trabajo'], inplace=True)
    df = normalizar(df)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder

def ingenieriaDeFeaturesBoosting(df:pd.DataFrame):

    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    
    df,me = meanEncoding(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    
    df.drop(columns= ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada'], inplace=True)
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder, me 

def ingenieriaDeFeaturesCategoricalNB(df:pd.DataFrame):
    categories = ['estado_marital', 'genero', 'trabajo', 'categoria_de_trabajo']
    df = codificacionOrdinal(df, categories)
    df.drop(columns = ['religion', 'edad', 'horas_trabajo_registradas', 'barrio', 'educacion_alcanzada',  
                       'rol_familiar_registrado', 'ganancia_perdida_declarada_bolsa_argentina',   
                       'anios_estudiados'], inplace = True)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)
    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder 
def ingenieriaDeFeaturesCategoricalNB2(df:pd.DataFrame):
    categories = ['estado_marital', 'genero', 'trabajo', 'categoria_de_trabajo']
    df,me = meanEncoding(df,categories)
    df.drop(columns = ['religion', 'edad', 'horas_trabajo_registradas', 'barrio', 'educacion_alcanzada',  
                       'rol_familiar_registrado', 'ganancia_perdida_declarada_bolsa_argentina',   
                       'anios_estudiados'], inplace = True)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)
    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder, me
    
        
def ingenieriaDeFeaturesGaussianNB(df:pd.DataFrame):
    
    df.drop(columns = ['religion', 'horas_trabajo_registradas', 'barrio', 'educacion_alcanzada', 
                       'estado_marital', 'genero', 'trabajo', 'categoria_de_trabajo', 
                       'rol_familiar_registrado'], inplace = True)
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)
    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder 
   
def ingenieriaDeFeauturesRegresion1(df:pd.DataFrame):
    
    X, y, df, label_encoder = ingenieriaDeFeaturesKnn(df)
    
    return X, y, df, label_encoder

def ingenieriaDeFeauturesRegresion2(df:pd.DataFrame):
    
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    
    df,me = meanEncoding(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada'], inplace=True)
    
    df = normalizar(df)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder,me

def ingenieriaDeFeaturesKnn2(df:pd.DataFrame):
    
    categories = ['categoria_de_trabajo', 'estado_marital', 'genero', 'trabajo', 'rol_familiar_registrado']
    df, me = meanEncoding(df, categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada'], inplace=True)
    
    df = normalizar(df)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder,me

def ingenieriaDeFeaturesRedes(df:pd.DataFrame):
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero', 'trabajo']
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada',
                    'rol_familiar_registrado', 'anios_estudiados'], inplace=True)
    df = normalizar(df)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder


def ingenieriaDeFeaturesRedes2(df:pd.DataFrame):
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero', 'trabajo']
    df, me = meanEncoding(df, categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada',
                    'anios_estudiados', 'rol_familiar_registrado'], inplace=True)
    #df = pasarAFloat32(df)
    df = normalizar(df)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder,me
    
    
def prepararSetDeHoldOutRedes(df):
    categories = ['categoria_de_trabajo', 'estado_marital', 'genero', 'trabajo']
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada',
                    'rol_familiar_registrado', 'anios_estudiados','id','representatividad_poblacional'], inplace=True)
    df = normalizar(df)
    return df
    

def prepararSetDeHoldOutArbol(df):
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
              'rol_familiar_registrado', 'trabajo']
    
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    
    df.drop(columns= ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','anios_estudiados'],inplace=True)
    return df

def prepararSetDeHoldOutKNN(df, meanEncoding):
    categories = ['categoria_de_trabajo', 'estado_marital', 'genero', 'trabajo', 'rol_familiar_registrado']
    df = completarConMeanEncoding(df, meanEncoding)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada',
                     'id','representatividad_poblacional'], inplace=True)
    
    df = normalizar(df)
    return df
    

def prepararSetDeHoldOutBoosting(df,meanEncoding):
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    
    df = prepararSetDeEntrenamiento(df)
    df = completarConMeanEncoding(df,meanEncoding)
    df = ordinalEncodingEducacionAlcanzada(df)
    
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','id','representatividad_poblacional'], inplace=True)
 
    return df

def prepararSetDeHoldOutRegresion(df):
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    
    df = prepararSetDeEntrenamiento(df)
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','id','representatividad_poblacional'], inplace=True)
    
    df = normalizar(df)

    return df
    
def prepararSetDeHoldOutCategoricalNB(df):
    categories = ['estado_marital', 'genero', 'trabajo', 'categoria_de_trabajo']
    df = codificacionOrdinal(df, categories)
    df.drop(columns = ['religion', 'edad', 'horas_trabajo_registradas', 'barrio', 'educacion_alcanzada',  
                       'rol_familiar_registrado', 'ganancia_perdida_declarada_bolsa_argentina',   
                       'anios_estudiados', 'id', 'representatividad_poblacional'], inplace = True)
    return df

def prepararSetDeHoldOutGaussianNB(df):
    df.drop(columns = ['religion', 'horas_trabajo_registradas', 'barrio', 'educacion_alcanzada',
                       'estado_marital', 'genero', 'trabajo', 'categoria_de_trabajo', 
                       'rol_familiar_registrado',  'id', 'representatividad_poblacional'], inplace = True)
    return df