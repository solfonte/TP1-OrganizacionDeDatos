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
    #hay q borrar esta funcion
    validation_df = prepararSetDeEntrenamiento(validation_df)
    
    validation_df.drop(columns = ['representatividad_poblacional'])
    
    return validation_df


def oneHotEncodingArbol1(df:pd.DataFrame):
    
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
              'rol_familiar_registrado', 'trabajo']
    
    df = pd.get_dummies(df, drop_first = True, columns = categories)
    
    return df

def oneHotEncodingArbol2(df:pd.DataFrame):
    
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']

    
    df = pd.get_dummies(df, drop_first = True, columns = categories)
    
    return df

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

def ingenieriaDeFeaturesArboles1(df:pd.DataFrame):
    
    """Hace las transformaciones de datos necesarias para entrenar al arbol de decision."""
    
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
              'rol_familiar_registrado', 'trabajo']
    
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    
    df.drop(columns=
      ['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada','anios_estudiados'],inplace=True)
   
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])# se saca la variable target para evitar un leak en el entrenamiento
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder

def reducirTrabajos(df:pd.DataFrame):
    #para agregar a la explcacion: juntamos al ejercito, domestico y limpiador con "otros" ya que son
    #similares en cuanto a poder adquisitivo y tienen pocos encuestados 
    df['trabajo'] = df['trabajo'].replace('limpiador', 'otros')
    df['trabajo'] = df['trabajo'].replace('servicio_domestico', 'otros')
    df['trabajo'] = df['trabajo'].replace('ejercito', 'otros')
    return df

def reducirCategorias(df:pd.DataFrame):
    #estos dos los juntamos prque tienen distribucion parecida de poder adquisitivo
    #los de sin trabajo los vimos que estaban correlacionados en 1 con los No respondio
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].replace('empleado_municipal', 'empleadao_estatal')
    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].replace('empleado_provincial', 'empleadao_estatal')

    return df

def reducirEstadoMarital(df:pd.DataFrame):
    #estos dos los juntamos prque tienen distribucion parecida de poder adquisitivo
    #los de sin trabajo los vimos que estaban correlacionados en 1 con los No respondio
    df['estado_marital'] = df['estado_marital'].replace('divorciado', 'sin_matrimonio')
    df['estado_marital'] = df['estado_marital'].replace('pareja_no_presente', 'sin_matrimonio')
    df['estado_marital'] = df['estado_marital'].replace('separado', 'sin_matrimonio')
    df['estado_marital'] = df['estado_marital'].replace('viudo_a', 'sin_matrimonio')
    
    df['estado_marital'] = df['estado_marital'].replace('matrimonio_civil', 'matrimonio')
    df['estado_marital'] = df['estado_marital'].replace('matrimonio_militar', 'matrimonio')

    return df


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
   

def oneHotEncodingCodificar(df:pd.DataFrame,categories):
    df = pd.get_dummies(df, drop_first = True, columns = categories)
    return df

def normalizar(df:pd.DataFrame):
    return (df - df.mean()) / df.std()

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
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada',    'rol_familiar_registrado', 'categoria_de_trabajo'], inplace=True)
    df = normalizar(df)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.tiene_alto_valor_adquisitivo)

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
    y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

    return X, y, df, label_encoder


def meanEncoding(df:pd.DataFrame,categories):
    meanEncodedCategories = {}
    for cat in categories: 
        categoriaEncodeada = df.groupby([cat])['tiene_alto_valor_adquisitivo'].mean().to_dict()
        df[cat] =  df[cat].map(categoriaEncodeada)
        meanEncodedCategories[cat] = categoriaEncodeada
        
    return df,meanEncodedCategories

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

def codificacionOrdinal(df, categories):
    encoder = OrdinalEncoder()
    df[categories] = encoder.fit_transform(df[categories])
    return df

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

    X = df.drop(columns=['tiene_alto_valor_adquisitivo'])# se saca la variable target para evitar un leak en el   entrenamiento
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

def prepararSetDeHoldOutKNN(df):
    categories = [ 'categoria_de_trabajo', 'estado_marital', 'genero',
                  'rol_familiar_registrado', 'trabajo']
    df = oneHotEncodingCodificar(df,categories)
    df = ordinalEncodingEducacionAlcanzada(df)
    df.drop(columns=['religion','horas_trabajo_registradas','edad','barrio','educacion_alcanzada'], inplace=True)
    
    df = normalizar(df)
    return df
    
def completarConMeanEncoding(df,meanEncoding):
    
    for cat in meanEncoding.keys():
        df[cat] =  df[cat].map(meanEncoding[cat])
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
def prepararHoldOut(df):
    df.fillna(np.nan, inplace = True)

    df['categoria_de_trabajo'] = df['categoria_de_trabajo'].replace(np.nan, 'No respondio')
    df['trabajo'] = df['trabajo'].replace(np.nan, 'No respondio')
    df['barrio'] = df['barrio'].replace(np.nan, 'Otro Barrio')
    
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