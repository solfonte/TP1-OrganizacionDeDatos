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


def ingenieriaDeFeaturesOH(df:pd.DataFrame,categoriasCodificar,categoriasEliminar):
    df = oneHotEncodingCodificar(df,categoriasCodificar)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns = categoriasEliminar,inplace=True)
    return df

def finalizarIngenieriaDeFeatures(df:pd.DataFrame):
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder

def encodearEducacion(educacion):
    if educacion.find("1-4_grado") >= 0:
        return 1
    elif educacion.find("5-6_grado") >= 0:
        return 2
    elif educacion.find("7-8_grado") >= 0:
        return 3
    elif educacion.find("9_grado") >= 0:
        return 4
    if educacion.find("1_anio") >= 0:
        return 5
    elif educacion.find("2_anio") >= 0:
        return 6
    elif educacion.find("3_anio") >= 0:
        return 7
    elif educacion.find("4_anio") >= 0:
        return 8
    if educacion.find("5_anio") >= 0:
        return 9
    elif educacion.find("universidad_1_anio") >= 0:
        return 10
    elif educacion.find("universidad_2_anio") >= 0:
        return 11
    elif educacion.find("universidad_3_anio") >= 0:
        return 12
    elif educacion.find("universidad_4_anio") >= 0:
        return 13
    elif educacion.find("universidad_5_anio") >= 0:
        return 14
    elif educacion.find("universidad_6_anio") >= 0:
        return 15
    else:
        return 0

    
def ordinalEncodingEducacionAlcanzada(df:pd.DataFrame):
 
    df['educacion_alcanzada_encoded'] = 0
    df['educacion_alcanzada_encoded'] = df['educacion_alcanzada'].apply(encodearEducacion)
    return df

def reducirTrabajos(df:pd.DataFrame):
    df['trabajo'] = df['trabajo'].replace('limpiador', 'otros')
    df['trabajo'] = df['trabajo'].replace('servicio_domestico', 'otros')
    df['trabajo'] = df['trabajo'].replace('ejercito', 'otros')
    return df

def reducirCategorias(df:pd.DataFrame):
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].replace('empleado_municipal', 'empleadao_estatal')
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].replace('empleado_provincial', 'empleadao_estatal')

    return df

def reducirEstadoMarital(df:pd.DataFrame):
    df['estado_marital'] = df['estado_marital'].replace('divorciado', 'sin_matrimonio')
    df['estado_marital'] = df['estado_marital'].replace('pareja_no_presente', 'sin_matrimonio')
    df['estado_marital'] = df['estado_marital'].replace('separado', 'sin_matrimonio')
    df['estado_marital'] = df['estado_marital'].replace('viudo_a', 'sin_matrimonio')
    
    df['estado_marital'] = df['estado_marital'].replace('matrimonio_civil', 'matrimonio')
    df['estado_marital'] = df['estado_marital'].replace('matrimonio_militar', 'matrimonio')

    return df
   
def oneHotEncodingCodificar(df:pd.DataFrame,categories):
    df = pd.get_dummies(df, drop_first = True, columns = categories)
    return df

def normalizar(df:pd.DataFrame):
    return (df - df.mean()) / df.std()

def meanEncoding(df:pd.DataFrame,categories):
    meanEncodedCategories = {}
    for cat in categories: 
        categoriaEncodeada = df.groupby([cat])['tiene_alto_valor_adquisitivo'].mean().to_dict()
        df[cat] =  df[cat].map(categoriaEncodeada)
        meanEncodedCategories[cat] = categoriaEncodeada
        
    return df,meanEncodedCategories

def codificacionOrdinal(df, categories):
    encoder = OrdinalEncoder()
    df[categories] = encoder.fit_transform(df[categories])
    return df

def completarConMeanEncoding(df,meanEncoding):
    
    for cat in meanEncoding.keys():
        df[cat] =  df[cat].map(meanEncoding[cat])
    return df

def ingenieriaDeFeaturesME(df:pd.DataFrame,categoriasCodificar,categoriasEliminar):
    df,me = meanEncoding(df,categoriasCodificar)
    df = ordinalEncodingEducacionAlcanzada(df)
    
    df.drop(columns = categoriasEliminar, inplace=True)
    
    return df,me