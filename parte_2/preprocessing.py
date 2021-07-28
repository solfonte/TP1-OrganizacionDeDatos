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

def prepararSet(df:pd.DataFrame): 
    df.fillna(np.nan, inplace = True)
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].replace(np.nan, 'No respondio')
    df['trabajo'] = df['trabajo'].replace(np.nan, 'No respondio')
    df['barrio'] = df['barrio'].replace(np.nan, 'Otro Barrio')
    df['estado_marital'] = df['estado_marital'].replace(np.nan, 'No respondio')
    return df


def ingenieriaDeFeaturesArboles1(df:pd.DataFrame):  
    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
              'rol_familiar_registrado', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada']
    df = ingenieriaDeFeaturesOH(df,categoriasCodificar,categoriasEliminar) 
    
    return finalizarIngenieriaDeFeatures(df)

def ingenieriaDeFeaturesArboles2(df:pd.DataFrame):
    
    df = reducirTrabajos(df)
    df = reducirCategorias(df)
    df = reducirEstadoMarital(df)
    
    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','anios_estudiados']

    df = ingenieriaDeFeaturesOH(df,categoriasCodificar,categoriasEliminar)
    
    return finalizarIngenieriaDeFeatures(df) 

def ingenieriaDeFeaturesVariablesNormalizadas(df:pd.DataFrame):
    
    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']

    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada']

    df = ingenieriaDeFeaturesOH(df,categoriasCodificar,categoriasEliminar)
    df = normalizar(df)
    
    return finalizarIngenieriaDeFeatures(df)

def ingenieriaDeFeaturesSVM(df:pd.DataFrame):
    
    categoriasCodificar = ['estado_marital', 'genero', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','rol_familiar_registrado', 'categoria_de_trabajo']
    
    df = ingenieriaDeFeaturesOH(df,categoriasCodificar,categoriasEliminar)
    df = normalizar(df) 

    return finalizarIngenieriaDeFeatures(df)

def ingenieriaDeFeaturesBoosting(df:pd.DataFrame):

    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada']
    
    df,me = ingenieriaDeFeaturesME(df,categoriasCodificar,categoriasEliminar)
    X, y, df, label_encoder = finalizarIngenieriaDeFeatures(df)
    
    return X, y, df, label_encoder, me 

def ingenieriaDeFeaturesCategoricalNB(df:pd.DataFrame):
    
    categories = ['estado_marital', 'genero', 'trabajo', 'categoria_de_trabajo']
    df = codificacionOrdinal(df, categories)
    df.drop(columns = ['religion', 'edad', 'horas_trabajo_registradas', 'barrio', 'educacion_alcanzada',  
                       'rol_familiar_registrado', 'ganancia_perdida_declarada_bolsa_argentina',   
                       'anios_estudiados'], inplace = True)
    
    return finalizarIngenieriaDeFeatures(df)

def ingenieriaDeFeaturesCategoricalNB2(df:pd.DataFrame):
    categoriasCodificar = ['estado_marital', 'genero', 'trabajo', 'categoria_de_trabajo']
    categoriasEliminar = ['religion', 'edad', 'horas_trabajo_registradas', 'barrio', 'educacion_alcanzada',  
                       'rol_familiar_registrado', 'ganancia_perdida_declarada_bolsa_argentina',   
                       'anios_estudiados']
    
    df,me = ingenieriaDeFeaturesME(df,categoriasCodificar,categoriasEliminar)
    X, y, df, label_encoder = finalizarIngenieriaDeFeatures(df)
    
    return X, y, df, label_encoder, me
    
        
def ingenieriaDeFeaturesGaussianNB(df:pd.DataFrame):
    categoriasCodificar = ['estado_marital', 'genero', 'trabajo', 'categoria_de_trabajo', 'barrio',
                            'rol_familiar_registrado']
    categoriasEliminar = ['religion', 'educacion_alcanzada']
    
    df,me = ingenieriaDeFeaturesME(df,categoriasCodificar,categoriasEliminar)
    
    X, y, df, label_encoder = finalizarIngenieriaDeFeatures(df)
    
    return X, y, df, label_encoder,me

def ingenieriaDeFeauturesVariablesNormalizadasME(df:pd.DataFrame):
    
    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada']
    
    df,me = ingenieriaDeFeaturesME(df,categoriasCodificar,categoriasEliminar)
    df = normalizar(df)
    X, y, df, label_encoder = finalizarIngenieriaDeFeatures(df)
    
    return X, y, df, label_encoder,me


def ingenieriaDeFeaturesRedes(df:pd.DataFrame):
    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada',
                    'rol_familiar_registrado', 'anios_estudiados']
    
    df = ingenieriaDeFeaturesOH(df,categoriasCodificar,categoriasEliminar)
    df = normalizar(df)
    
    return finalizarIngenieriaDeFeatures(df)


def ingenieriaDeFeaturesRedes2(df:pd.DataFrame):
    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada',
                    'anios_estudiados', 'rol_familiar_registrado']
    
    df,me = ingenieriaDeFeaturesME(df,categoriasCodificar,categoriasEliminar)
    df = normalizar(df)
    X, y, df, label_encoder = finalizarIngenieriaDeFeatures(df)
    
    return X, y, df, label_encoder,me
    
    
def prepararSetDeHoldOutRedes(df):
    categoriasCodificar = ['categoria_de_trabajo', 'estado_marital', 'genero', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada',
                    'rol_familiar_registrado', 'anios_estudiados','id','representatividad_poblacional']
    
    df = ingenieriaDeFeaturesOH(df,categoriasCodificar,categoriasEliminar)
    df = normalizar(df)
    return df
    

def prepararSetDeHoldOutArbol(df):
    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
              'rol_familiar_registrado', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','id','representatividad_poblacional']
    df = prepararSet(df)
    df = ingenieriaDeFeaturesOH(df,categoriasCodificar,categoriasEliminar) 
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
    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
              'rol_familiar_registrado', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','id','representatividad_poblacional']
    df = prepararSet(df)
    df = ingenieriaDeFeaturesOH(df,categoriasCodificar,categoriasEliminar) 

    return df

def prepararSetDeHoldOutRegresion(df):
    categoriasCodificar = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    categoriasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','id','representatividad_poblacional']
    df = prepararSet(df)

    df = ingenieriaDeFeaturesOH(df,categoriasCodificar,categoriasEliminar)
    df = normalizar(df)

    return df
    
def prepararSetDeHoldOutCategoricalNB(df):
    categories = ['estado_marital', 'genero', 'trabajo', 'categoria_de_trabajo']
    df = codificacionOrdinal(df, categories)
    df.drop(columns = ['religion', 'edad', 'horas_trabajo_registradas', 'barrio', 'educacion_alcanzada',  
                       'rol_familiar_registrado', 'ganancia_perdida_declarada_bolsa_argentina',   
                       'anios_estudiados', 'id', 'representatividad_poblacional'], inplace = True)
    return df

def prepararSetDeHoldOutGaussianNB(df, meanEncoding):
   
    df = completarConMeanEncoding(df, meanEncoding)
    df.drop(columns = ['religion', 'horas_trabajo_registradas', 'barrio', 'educacion_alcanzada',
                        'genero', 'categoria_de_trabajo',  'id', 'representatividad_poblacional'], inplace = True)
    return df

def prepararSetDeHoldOutSvm(df):
    categories = ['estado_marital', 'genero', 'trabajo']
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    columnasEliminar = ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','rol_familiar_registrado', 'categoria_de_trabajo', 'id','representatividad_poblacional']
    df.drop(columns = columnasEliminar, inplace=True)
    df = normalizar(df) 
    return df