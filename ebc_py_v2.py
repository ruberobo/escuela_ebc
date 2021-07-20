# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import seaborn as sns
import statistics as stat
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

año_actual=2021
año_anterior=2020

ruta=r"C:\Users\GOTSA\Documents\ruben\ebc\CE-EBC\\"

#leo archivo historico
nombre_archivo_historico='Historico-materias.csv'
historico= pd.read_csv(ruta+nombre_archivo_historico, skiprows=[1])

#borro nulos
historico.dropna(axis=0, inplace=True)

#función formato
def formato(x):
    try:
        return str(int(x)).strip()
    except:
        return x
    
#corrijo formato
historico['PIDM']=historico['PIDM'].apply(lambda x: formato(x))
historico['CURSO']=historico['CURSO'].apply(lambda x: formato(x))
historico['PERIODO-COMPRA']=historico['PERIODO-COMPRA'].apply(lambda x:formato(x))
historico['PLAN']=historico['PLAN'].apply(lambda x:formato(x))

historico_sin_bajas=historico[(historico['CODIGO ESTATUS MATERIA']=='RE')|(historico['CODIGO ESTATUS MATERIA']=='RW')].reset_index(drop=True)
historico_con_bajas=historico[['PIDM','CURSO', 'CODIGO ESTATUS MATERIA', 'PERIODO-COMPRA','MODALIDAD-MATERIA']]
historico_con_bajas['id']=historico_con_bajas['PIDM']+'-'+historico_con_bajas['CURSO']

historico_2=historico_sin_bajas[['PIDM','CURSO', 'CODIGO ESTATUS MATERIA', 'PERIODO-COMPRA','MODALIDAD-MATERIA']]

historico_2['id']=historico_2['PIDM']+'-'+historico_2['CURSO']




#leo archivo población
nombre_archivo_poblacion='Poblacion.csv'
población= pd.read_csv(ruta+nombre_archivo_poblacion, skiprows=[1])
#borro nulos
población.dropna(axis=0, inplace=True)

#aplico formato
población['PIDM']=población['PIDM'].apply(lambda x: formato(x))
población['PERIODO-INICIO-CURRICULA']=población['PERIODO-INICIO-CURRICULA'].apply(lambda x: formato(x))
población['PERIODO-ULTIMO-CURSADO']=población['PERIODO-ULTIMO-CURSADO'].apply(lambda x: formato(x))
población['PERIODO-TIPOALUM-HISTORICO']=población['PERIODO-TIPOALUM-HISTORICO'].apply(lambda x: formato(x))
población['PIDM']=población['PIDM'].apply(lambda x: formato(x))

población_sin_graduados= población[['PIDM','MATRICULA','ESTATUS-INSCRIPCION-ALUMNO', 'PERIODO-INICIO-CURRICULA',
       'PERIODO-ULTIMO-CURSADO', 'PERIODO-TIPOALUM-HISTORICO','NOMBRE ALUMNO', 'TOTAL MATERIAS CURSADAS BANNER',
                        'CAMPUS','PROGRAMA']][población['TOTAL MATERIAS CURSADAS BANNER']<42]

población_con_graduados= población[['PIDM','MATRICULA','ESTATUS-INSCRIPCION-ALUMNO', 'PERIODO-INICIO-CURRICULA',
       'PERIODO-ULTIMO-CURSADO', 'PERIODO-TIPOALUM-HISTORICO','NOMBRE ALUMNO', 'TOTAL MATERIAS CURSADAS BANNER',
                        'CAMPUS','PROGRAMA']][población['TOTAL MATERIAS CURSADAS BANNER']<=42]



#leo archivo de kardex
nombre_archivo_kardex='kardex.csv'
kardex=pd.read_csv(ruta+nombre_archivo_kardex,skiprows=[1])
#aplico formato
kardex['PIDM']=kardex['PIDM'].apply(lambda x: formato(x))
kardex['CURSO']=kardex['CURSO'].apply(lambda x: formato(x))
kardex['PERIODO-MATERIA']=kardex['PERIODO-MATERIA'].apply(lambda x: formato(x))
kardex['PLAN']=kardex['PLAN'].apply(lambda x: formato(x))

#lleno nulos
kardex[['Materia-Cursadas','MATERIAS-EN-CURSO']]=kardex[['Materia-Cursadas','MATERIAS-EN-CURSO']].fillna(0)

kardex['CALIFICACION']=np.where(kardex['MATERIAS-EN-CURSO']==1,0,kardex['CALIFICACION'])
kardex['id']=kardex['PIDM']+'-'+kardex['CURSO']

kardex_2=kardex[(kardex['Materia-Cursadas']!=0)|(kardex['MATERIAS-EN-CURSO']!=0)]

kardex_materias_terminadas=kardex_2[['id','semestre','MODALIDA-MATERIA','Materia-Cursadas','MATERIAS-EN-CURSO','MATERIAS DEL CURSO', 'CALIFICACION', 'CAMPUS-MATERIAS','PLAN']]

#incluyo materias que no se terminaron
kardex_todas_materias=kardex[['id','semestre','MODALIDA-MATERIA','Materia-Cursadas','MATERIAS-EN-CURSO','MATERIAS DEL CURSO', 'CALIFICACION', 'CAMPUS-MATERIAS','PLAN']]



#leo status
nombre_archivo_estatus='Estatus_alumno.txt'
estatus=pd.read_csv(ruta+nombre_archivo_estatus, delimiter="\t" )





#junto tres tablas
df1=pd.merge(historico_2,kardex_materias_terminadas, on='id', how="inner")

alumnos=población_con_graduados.drop_duplicates(subset=['PIDM'], keep='last').reset_index(drop=True)

alumnos_2=pd.merge(alumnos,estatus, on='MATRICULA', how="inner")

materias=pd.merge(df1,alumnos_2, on='PIDM', how="inner")

#lleno nulos
materias[['MODALIDA-MATERIA','CAMPUS-MATERIAS']]=materias[['MODALIDA-MATERIA','CAMPUS-MATERIAS']].fillna('no disponible')


tabla=materias.groupby(['MATRICULA',]).agg({'CALIFICACION':'mean', 'semestre':'max', 'PIDM':'count'})
tabla['MATRICULA']=tabla.index
tabla_2=tabla.rename(columns={'CALIFICACION':'promedio','semestre':'ultimo semestre','PIDM':'materias pagadas'})
tabla_3=tabla_2.reset_index(drop=True)

tabla_4=pd.merge(tabla_3,alumnos_2, on='MATRICULA', how="inner")


codigo_estatus=[]
for i in tabla_4['Estatus Alumno']:
    if i =='ACTIVO':
        codigo_estatus.append(1)
    elif i=='INACTIVO':
        codigo_estatus.append(2)
    else:
        codigo_estatus.append(0)
        
tabla_4['codigo_estatus']=codigo_estatus    

tabla_4['estatus_alumno']=tabla_4['Estatus Alumno']    


tabla_5=pd.get_dummies(tabla_4, columns=['Estatus Alumno'])
tabla_5['terminó último periodo']=np.where(tabla_5['PERIODO-ULTIMO-CURSADO']==tabla_5['PERIODO-TIPOALUM-HISTORICO'],1,0)
print(tabla_5.columns)

#funciones para limpiar

def año_inicial(x):
    z=re.findall(r'^\d{4}', x)
    return int(z[0])

def periodo_inicial(x):
    z=re.findall(r'\d{2}$', x)
    return int(z[0])

tabla_5['año inicial']= tabla_5['PERIODO-INICIO-CURRICULA'].apply(año_inicial)
tabla_5['periodo inicial']=tabla_5['PERIODO-INICIO-CURRICULA'].apply(periodo_inicial)
tabla_5['año final']= tabla_5['PERIODO-ULTIMO-CURSADO'].apply(año_inicial)
tabla_5['periodo final']=tabla_5['PERIODO-ULTIMO-CURSADO'].apply(periodo_inicial)

tabla_6=tabla_5[['Estatus Alumno_ACTIVO','promedio', 'ultimo semestre', 'TOTAL MATERIAS CURSADAS BANNER', 'materias pagadas',
                 'terminó último periodo', 'año inicial','periodo inicial','año final', 'periodo final']]


#modelo nuevos alumnos
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


nuevos_alumnos=tabla_5.groupby('año inicial').agg({'MATRICULA':'count'})
nuevos_alumnos['año_inicial']=nuevos_alumnos.index

y_nuevos_alumnos=nuevos_alumnos[['MATRICULA']][9:-1]
x_nuevos_alumnos=nuevos_alumnos[['año_inicial']][9:-1]

poly_nuevos=make_pipeline(PolynomialFeatures(2),LinReg())

poly_nuevos.fit(x_nuevos_alumnos,y_nuevos_alumnos)

nuevos_alumnos_2021=int(poly_nuevos.predict([[2021]]))

lista_campus_matricula=list(tabla_5['CAMPUS'].value_counts().index)

#for por campus
predicciones_nuevos_alumnos=[]
nuevos_alumnos_campus=[]
for i in lista_campus_matricula:
    df=tabla_5[tabla_5['CAMPUS']==i]
    grupo_alumnos=df.groupby('año inicial').agg({'MATRICULA':'count'})
    grupo_alumnos['año_inicial']=grupo_alumnos.index
    
    #filtro año
    grupo_alumnos_2=grupo_alumnos[grupo_alumnos['año_inicial']>2012]
    nuevos_alumnos_campus.append(grupo_alumnos_2)
  
    y_grupo_alumnos=grupo_alumnos_2[['MATRICULA']][:-1]
    x_grupo_alumnos=grupo_alumnos_2[['año_inicial']][:-1]
    poly_nuevos=make_pipeline(PolynomialFeatures(2),LinReg())
    
    try:
        poly_nuevos.fit(x_grupo_alumnos,y_grupo_alumnos)
        
    except:
        pass
    
    nuevos_alumnos_2021=int(poly_nuevos.predict([[2021]]))
    
    if nuevos_alumnos_2021>0:
        predicciones_nuevos_alumnos.append([i,nuevos_alumnos_2021])
    else:
        predicciones_nuevos_alumnos.append([i,0])

nuevos_historico=pd.pivot_table(tabla_5, columns= 'año inicial', index='CAMPUS', values='MATRICULA', aggfunc='count').fillna(0)
nuevos_historico['campus']=nuevos_historico.index

nuevos_alumnos_campus=pd.DataFrame(predicciones_nuevos_alumnos, columns=['campus','pronostico_alumnos_2021'])
tabla_nuevos_alumnos=pd.merge(nuevos_alumnos_campus,nuevos_historico, on='campus', how="inner")

#exporto
tabla_nuevos_alumnos.to_csv('out\\nuevos_alumnos.csv', sep=',')


#por carrera
historico_carrera=pd.pivot_table(tabla_5, columns= ['año inicial','PERIODO-INICIO-CURRICULA'], index=['CAMPUS','PROGRAMA'], values='MATRICULA', aggfunc='count').fillna(0)
historico_prob=pd.DataFrame(index=historico_carrera[2020].index, columns=historico_carrera[2020].columns)

#valor total
suma_lista=[]
for i in historico_carrera[2020].columns:
    x=historico_carrera[2020][i].sum()
    suma_lista.append(x)
    
    
suma_2020=sum(suma_lista)

for i in historico_carrera[2020].columns:
    indice=i
    lista_auxiliar=[]
    for j in range(len(historico_carrera)):
        x=float(historico_carrera[2020][indice][j]/suma_2020)
        lista_auxiliar.append(x)
    historico_prob[indice]=lista_auxiliar
    
#funcion corrige periodo
def periodo(x):
    return x.replace(str(año_anterior),str(año_actual))


pronostico_alumnos_campus_carrera=pd.DataFrame(index=historico_carrera[2020].index, columns=historico_carrera[2020].columns)

tabla_nuevos_alumnos['pronostico_alumnos_2021'].sum()


for i in pronostico_alumnos_campus_carrera:
    indice=i
    pronostico_alumnos_campus_carrera[indice]=round(tabla_nuevos_alumnos['pronostico_alumnos_2021'].sum()*historico_prob[i])

nuevos_nombres=[]
for i in pronostico_alumnos_campus_carrera.columns:
    nuevos_nombres.append(periodo(i))
    
renombro=dict(zip(pronostico_alumnos_campus_carrera.columns, nuevos_nombres))

pronostico_alumnos_campus_carrera.rename(columns=renombro, inplace=True)

#exporto,
pronostico_alumnos_campus_carrera.to_csv('out\\nuevos_alumnos_2021_campus_carrera_periodo.csv', sep=',')

#exporto
historico_carrera.to_csv('out\\historico_carrera.csv', sep=',')


#modelo egresados

tabla_5_egresados=tabla_5[tabla_5['TOTAL MATERIAS CURSADAS BANNER']==42]
egresados_historico=pd.pivot_table(tabla_5_egresados, columns= 'año final', index='CAMPUS', values='MATRICULA', aggfunc='count').fillna(0)

predicciones_graduados=[]
df=tabla_5[tabla_5['TOTAL MATERIAS CURSADAS BANNER']==42]
for i in lista_campus_matricula:
    df_2=df[df['CAMPUS']==i]
    grupo_alumnos=df_2.groupby('año final').agg({'MATRICULA':'count'})
    grupo_alumnos['año_final']=grupo_alumnos.index
    
    y_grupo_alumnos=grupo_alumnos[['MATRICULA']][:-1]
    x_grupo_alumnos=grupo_alumnos[['año_final']][:-1]
    
    try:
        poly_nuevos=make_pipeline(PolynomialFeatures(2),LinReg())
        poly_nuevos.fit(x_grupo_alumnos,y_grupo_alumnos)
        
        nuevos_graduado_2021=int(poly_nuevos.predict([[2021]]))
        
    except:
        nuevos_graduado_2021=0
    
    if nuevos_graduado_2021>0:
        predicciones_graduados.append([i,nuevos_graduado_2021])
    else:
        predicciones_graduados.append([i,0])
        
        

nuevos_egresados=pd.DataFrame(predicciones_graduados, columns=['campus','pronostico_graduados_2021'])

egresados_historico['campus']=egresados_historico.index

tabla_egresados=pd.merge(nuevos_egresados,egresados_historico, on='campus', how="inner")

historico_egresados=pd.pivot_table(tabla_5_egresados, columns= ['año final','PERIODO-ULTIMO-CURSADO'], index=['CAMPUS','PROGRAMA'], values='MATRICULA', aggfunc='count').fillna(0)

#exporto
historico_egresados.to_csv('out\historico_egresados.csv', sep=',')

#probabilidad
historico_egresado_prob=pd.DataFrame(index=historico_egresados[2020].index, columns=historico_egresados[2020].columns)

#valor total
suma_lista_e=[]
for i in historico_egresados[2020].columns:
    x=historico_egresados[2020][i].sum()
    suma_lista_e.append(x)
    
    
suma_egresados_2020=sum(suma_lista_e)


for i in historico_egresados[2020].columns:
    indice=i
    lista_auxiliar_2=[]
    for j in range(len(historico_egresados)):
        x=float(historico_egresados[2020][indice][j]/suma_egresados_2020)
        lista_auxiliar_2.append(x)
    historico_egresado_prob[indice]=lista_auxiliar_2
    
    
    
pronostico_egresados_campus_carrera=pd.DataFrame(index=historico_egresados[2020].index, columns=historico_egresados[2020].columns)


for i in pronostico_egresados_campus_carrera:
    indice=i
    pronostico_egresados_campus_carrera[indice]=round(tabla_egresados['pronostico_graduados_2021'].sum()*historico_egresado_prob[i])
    

pronostico_egresados_campus_carrera.sum().sum()


#exporto
pronostico_egresados_campus_carrera.to_csv('out\pronostico_egresados_campus_carrera.csv', sep=',')



         
                                                                                                                                    



#clusters alumnos

#aqui iban los clusters sin considerar la cartera

#exporto
#tabla_6.to_csv('out\clusters_alumnos.csv', sep=',')



#modelo de registros
df2=pd.merge(historico_con_bajas,kardex_todas_materias, on='id', how="inner")
materias_registradas=pd.merge(df2,(pd.merge(alumnos,estatus, on='MATRICULA', how="inner")), on='PIDM', how="inner")

materias_2=materias_registradas[['id','CURSO', 'PERIODO-COMPRA','CODIGO ESTATUS MATERIA','MATERIAS DEL CURSO','MODALIDAD-MATERIA','semestre','CALIFICACION','CAMPUS' ,'CAMPUS-MATERIAS', 'PLAN',
            'PERIODO-INICIO-CURRICULA','PERIODO-ULTIMO-CURSADO', 'PERIODO-TIPOALUM-HISTORICO','MATERIAS-EN-CURSO','TOTAL MATERIAS CURSADAS BANNER','Estatus Alumno','PROGRAMA']]


materias_2['terminó último periodo']=np.where(materias_2['PERIODO-ULTIMO-CURSADO']==materias_2['PERIODO-TIPOALUM-HISTORICO'],1,0)

#lleno nulos
materias_3=materias_2.fillna(0)

materias_3['materia inscrita']=np.where((materias_3['CODIGO ESTATUS MATERIA']=='RE')|(materias_3['CODIGO ESTATUS MATERIA']=='RW'),1,0)
materias_3['materia_baja']=np.where((materias_3['CODIGO ESTATUS MATERIA'].str.startswith('B')),1,0)
materias_3['materia terminada']=np.where((materias_3['CALIFICACION']!=0),1,0)


def campus(x):
    if x=='VIR':
        return 8
    else:
        try:
            z=re.findall(r'\d{2}$', x)
            return int(z[0])
        except:
            return x



materias_3['codigo_campus']=materias_3['CAMPUS-MATERIAS'].apply(campus)
materias_3['año compra']=materias_3['PERIODO-COMPRA'].apply(año_inicial)
materias_3['periodo compra']=materias_3['PERIODO-COMPRA'].apply(periodo_inicial)
materias_3['año inicial']=materias_3['PERIODO-INICIO-CURRICULA'].apply(año_inicial)
materias_3['periodo inicial']=materias_3['PERIODO-INICIO-CURRICULA'].apply(periodo_inicial)
materias_3['año final']=materias_3['PERIODO-ULTIMO-CURSADO'].apply(año_inicial)
materias_3['periodo final']=materias_3['PERIODO-ULTIMO-CURSADO'].apply(periodo_inicial)


def materias_sin_plan(x):
    z=re.findall(r'\d{3}$', x)
    return int(z[0])
 
materias_3['codigo_materias']=materias_3['CURSO'].apply(materias_sin_plan)
materias_3['PLAN']=materias_3['PLAN'].apply(lambda x: int(x))   

#Listas de campus
lista_campus_matricula=list(materias_3['CAMPUS'].value_counts().index)
lista_campus=list(materias_3['CAMPUS-MATERIAS'].value_counts().index)

campus_m_groupby=[]
for i in lista_campus_matricula:
    campus_m_groupby.append('CAMPUS_'+str(i))
    
campus_m_dict=dict(zip(campus_m_groupby,['sum']*len(campus_m_groupby)))


campus_groupby=[]
for i in lista_campus:
    campus_groupby.append('CAMPUS-MATERIAS_'+str(i))
      
campus_dict=dict(zip(campus_groupby,['sum']*len(campus_groupby)))



materias_4=pd.get_dummies(materias_3, columns=['MODALIDAD-MATERIA','periodo compra','MATERIAS DEL CURSO','CAMPUS-MATERIAS','CAMPUS'])



grupo_1=materias_4.groupby(['año compra']).agg({'id':'count','materia_baja':'sum','materia terminada':'sum',
                                           'MODALIDAD-MATERIA_PR':'sum', 'MODALIDAD-MATERIA_SP':'sum',
       'MODALIDAD-MATERIA_VR':'sum',})

grupo_campus=materias_4.groupby(['año compra']).agg(campus_dict)

grupo_campus_m=materias_4.groupby(['año compra']).agg(campus_m_dict)

grupo_por_año=pd.concat([grupo_1,grupo_campus,grupo_campus_m], axis=1)


grupo_por_año['año_compra']=grupo_por_año.index
grupo_por_año['año_compra']=grupo_por_año['año_compra'].apply(lambda x: int(x))
grupo_por_año.reset_index(drop=True, inplace=True)


#cambio de nombre
grupo_por_año.rename(columns={'id':'registros'}, inplace=True)


#regresion lineal
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

y=grupo_por_año[['registros']][:-1]
x=grupo_por_año[['año_compra']][:-1]

linreg=LinReg()
linreg.fit(x,y)

#prediccion
prediccion_2021=int(linreg.predict([[2021]]))

polyreg=make_pipeline(PolynomialFeatures(3),LinReg())
polyreg.fit(x,y)

pred_poly_2021=int(polyreg.predict([[2021]]))
pred_poly_2021

#predicción por campus MATERIAS
predicciones=[]
for i in lista_campus:
    indice='CAMPUS-MATERIAS_'+i
    y=grupo_por_año[[indice]][:-1]
    x=grupo_por_año[['año_compra']][:-1]
    polyreg=make_pipeline(PolynomialFeatures(3),LinReg())
    polyreg.fit(x,y)
    prediccion_2021=int(polyreg.predict([[2021]]))
    if prediccion_2021<=0:
        predicciones.append([i,0])
    else:
        predicciones.append([i,prediccion_2021])
        
prediccion_por_campus_materias=pd.DataFrame(data=predicciones,columns=['campus','materias'])

#PREDICCION ppor campus matricula
predicciones_m=[]
for i in lista_campus_matricula:
    indice='CAMPUS_'+i
    y=grupo_por_año[[indice]][:-1]
    x=grupo_por_año[['año_compra']][:-1]
    polyreg=make_pipeline(PolynomialFeatures(3),LinReg())
    polyreg.fit(x,y)
    prediccion_2021=int(polyreg.predict([[2021]]))
    if prediccion_2021<=0:
        predicciones_m.append([i,0])
    else:
        predicciones_m.append([i,prediccion_2021])
    
prediccion_por_campus_matriculas=pd.DataFrame(data=predicciones_m,columns=['campus','materias'])    
        







materias_5=pd.get_dummies(materias_3, columns=['MODALIDAD-MATERIA','periodo compra','año compra','CAMPUS-MATERIAS','CAMPUS'])

grupo_materias=materias_5.groupby(['codigo_materias']).agg({'id':'count','materia_baja':'sum','materia terminada':'sum',
                                           'MODALIDAD-MATERIA_PR':'sum', 'MODALIDAD-MATERIA_SP':'sum',
       'MODALIDAD-MATERIA_VR':'sum','año compra_2013':'sum',
       'año compra_2014':'sum', 'año compra_2015':'sum', 'año compra_2016':'sum',
       'año compra_2017':'sum', 'año compra_2018':'sum', 'año compra_2019':'sum',
       'año compra_2020':'sum', 'año compra_2021':'sum',})

grupo_materias.rename(columns={'id':'registros'}, inplace=True)
grupo_materias['codigo_materias']=grupo_materias.index

grupo_materias_modelo=grupo_materias[['codigo_materias','registros', 'materia_baja', 'materia terminada',
       'MODALIDAD-MATERIA_PR', 'MODALIDAD-MATERIA_SP', 'MODALIDAD-MATERIA_VR',
       'año compra_2013', 'año compra_2014', 'año compra_2015',
       'año compra_2016', 'año compra_2017', 'año compra_2018',
       'año compra_2019', 'año compra_2020','año compra_2021' ]]




#calculo materias año 2020
registros_2020=materias_3[materias_3['año compra']==2020]
materias_2020=pd.crosstab(registros_2020['MATERIAS DEL CURSO'], registros_2020['PERIODO-COMPRA'], margins=True)


total_2020=materias_2020['All']['All']
materias_prob=pd.DataFrame(index=materias_2020.index, columns=materias_2020.columns)

for i in list(materias_2020.columns):
    k=materias_2020[i]/total_2020
    materias_prob[i]=k
    
materias_2021_pred=pd.DataFrame(index=materias_2020.index, columns=materias_2020.columns)
periodos_2021=['202112','202125','202126','202127','202148','202149','202150']

for i in (list(materias_2020.columns)):
    k=round(materias_prob[i]*pred_poly_2021)
    materias_2021_pred[i]=k
    
materias_2021_pred.rename(columns={'202012':'202112','202025':'202125','202026':'202126','202027':'202127',
                                  '202048':'202148','202049':'202149','202050':'202150'}, inplace=True)


#agrego semestre y plan
materia_semestre=registros_2020.groupby(['CURSO']).agg({'semestre':'max', 'PLAN':'min'})
semestres=list(materia_semestre['semestre'].values)
semestres.append(0)

planes=list(materia_semestre['PLAN'].values)
planes.append(0)

#materias_2021_pred['plan']=planes
#materias_2021_pred['semestre']=semestres

#exporto
materias_2021_pred.to_csv('out\prediccion_codigo_materia_todos_los_campus.csv', sep=',')



#para campus materia y campus matricula
#funcion corrige periodo
def periodo(x):
    return x.replace('2020','2021')

#for por campus materias
lista_df_campus=[]
for i in lista_campus:
    df=materias_3[materias_3['CAMPUS-MATERIAS']==i]

    registros_2020=df[df['año compra']==2020]
    try:
        materias_2020=pd.crosstab(registros_2020['CURSO'], registros_2020['PERIODO-COMPRA'], margins=True)
    
        total_2020=materias_2020['All']['All']
    
        materias_prob=pd.DataFrame(index=materias_2020.index, columns=materias_2020.columns)
    
        for j in list(materias_2020.columns):
            k=materias_2020[j]/total_2020
            materias_prob[j]=k
        
        #columnas
        columnas=[]
        for j in list(materias_2020.columns):
            k=periodo(j)
            columnas.append(k)

        materias_2021_pred=pd.DataFrame(index=materias_2020.index, columns=materias_2020.columns)    
      
    
    
        for j in list(materias_2020.columns):
            k=round(materias_prob[j]*prediccion_por_campus_materias[prediccion_por_campus_materias['campus']==i]['materias'].values)
            materias_2021_pred[j]=k
            
        
        diccionario=dict(zip(materias_2020.columns, columnas))
        
        materias_2021_pred.rename(columns=diccionario, inplace=True)
    
        materia_semestre=registros_2020.groupby(['CURSO']).agg({'semestre':'max', 'PLAN':'min'})
    
        semestres=list(materia_semestre['semestre'].values)
        semestres.append(0)
    
        planes=list(materia_semestre['PLAN'].values)
        planes.append(0)
        materias_2021_pred['semestre']=semestres
        materias_2021_pred['plan']=planes
    
        lista_df_campus.append(materias_2021_pred)
        
    except:
        lista_df_campus.append(pd.DataFrame())


#exporto
for i in range(len(lista_campus)):
    nombre='prediccion_CAMPUS_MATERIA'+str(lista_campus[i])
    lista_df_campus[i].to_csv('out\CAMPUS_MATERIAS\\'+nombre+'.csv',sep=',')
    
    
#for por campus matRICULA
lista_df_campus_m=[]
for i in lista_campus_matricula:
    df=materias_3[materias_3['CAMPUS']==i]

    registros_2020=df[df['año compra']==2020]
    try:
        materias_2020=pd.crosstab(registros_2020['CURSO'], registros_2020['PERIODO-COMPRA'], margins=True)
        
    
        total_2020=materias_2020['All']['All']
        
    
    
        materias_prob=pd.DataFrame(index=materias_2020.index, columns=materias_2020.columns)
    
        for j in list(materias_2020.columns):
            k=materias_2020[j]/total_2020
            materias_prob[j]=k
        
        
        #columnas
        columnas=[]
        for j in list(materias_2020.columns):
            k=periodo(j)
            columnas.append(k)

        materias_2021_pred=pd.DataFrame(index=materias_2020.index, columns=materias_2020.columns)
        
    
        for j in list(materias_2020.columns):
            k=round(materias_prob[j]*prediccion_por_campus_matriculas[prediccion_por_campus_matriculas['campus']==i]['materias'].values)
            materias_2021_pred[j]=k
          
        
        diccionario=dict(zip(materias_2020.columns, columnas))
        
        materias_2021_pred.rename(columns=diccionario, inplace=True)
    
        materia_semestre=registros_2020.groupby(['CURSO']).agg({'semestre':'max', 'PLAN':'min'})
    
        semestres=list(materia_semestre['semestre'].values)
        semestres.append(0)
    
        planes=list(materia_semestre['PLAN'].values)
        planes.append(0)
        materias_2021_pred['semestre']=semestres
        materias_2021_pred['plan']=planes
    
        lista_df_campus_m.append(materias_2021_pred)
        
    except:
        lista_df_campus_m.append(pd.DataFrame())    


prediccion_por_campus_matriculas=pd.DataFrame(data=predicciones_m,columns=['campus','materias'])
prediccion_por_campus_matriculas


#exporto
for i in range(len(lista_campus_matricula)):
    nombre='prediccion_CAMPUS_matricula_'+str(lista_campus_matricula[i])
    lista_df_campus_m[i].to_csv('out\CAMPUS_MATRICULA\\'+nombre+'.csv',sep=',')





#cluster inscripciones
materias_modelo=materias_3[['semestre', 'materia inscrita','CALIFICACION','materia terminada',
       'codigo_campus', 'año compra', 'periodo compra', 'PLAN']]

materias_modelo['materia_virtual']=np.where(materias_3['MODALIDAD-MATERIA']=='VR',1,0)

Y_materias=materias_modelo['materia terminada']
X_materias=materias_modelo.drop(['materia terminada'],axis=1)

kmaterias=KMeans(n_clusters=3)
kmaterias.fit(X_materias)
y_pred_m=kmaterias.predict(X_materias)
#silhouette_score(X_materias,y_pred_m)

materias_final=materias_3

materias_final['cluster']=y_pred_m

maximo_m=materias_final['cluster'].value_counts().index[0]
medio_m=materias_final['cluster'].value_counts().index[1]
minimo_m=materias_final['cluster'].value_counts().index[2]

materias_final['grupo']=np.where(materias_final['cluster']==maximo_m,'incripcion_reciente',materias_final['cluster'])
materias_final['grupo']=np.where(materias_final['cluster']==medio_m,'inscripcion_historica',materias_final['grupo'])
materias_final['grupo']=np.where(materias_final['cluster']==minimo_m,'inscripcion_casi_graduado',materias_final['grupo'])


registro_terminado_actual=materias_final[(materias_final['materia terminada']==1)&(materias_final['año compra']>=2020)]
materias_terminadas_actual=registro_terminado_actual.groupby('CURSO').agg({'id':'count','CALIFICACION':'mean','PERIODO-COMPRA':'max'}).sort_values('id',ascending=False)
materias_terminadas_actual['CURSO']=materias_terminadas_actual.index

materias_terminadas_actual.reset_index(drop=True, inplace=True)
materias_terminadas_actual.rename(columns={'id':'registros_terminados_recientemiente','CALIFICACION':'promedio','PERIODO-COMPRA':'ultima compra'}, inplace=True)

materias_terminadas_actual_2=pd.merge(materias_terminadas_actual,materias_final,on='CURSO', how='inner')[['registros_terminados_recientemiente','promedio',
                                                                           'ultima compra','CURSO','MATERIAS DEL CURSO','PLAN']].drop_duplicates().reset_index(drop=True)




lista_materias=list(materias_terminadas_actual_2['CURSO'])

#modelo registros



#for grande
#for grande 
lista_pred=[]
for k in lista_materias:
    uno=materias_final[materias_final['CURSO']==k]
    año_uno=uno.groupby('año compra').agg({'id':'count'})
    año_uno['año']=año_uno.index
    
    uno_plan=list(uno['PLAN'].value_counts().index)[0]
    
    try:
        año_uno_x=año_uno[['año']][:-1]
        año_uno_y=año_uno[['id']][:-1]
        modelo_uno=make_pipeline(PolynomialFeatures(3),LinReg())
        modelo_uno.fit(año_uno_x, año_uno_y)
        prediccion=int(modelo_uno.predict([[2021]]))
    
        if prediccion>=0:
            pred_uno=prediccion
        else:
            pred_uno=0
    
    except:
        pred_uno=0
    
    Uno_año=pd.pivot_table(uno, columns= ['año compra','periodo compra'], index=['CAMPUS','PROGRAMA'], values='id', aggfunc='count').fillna(0)
    uno_prob=pd.DataFrame(index=Uno_año[2020].index,columns=Uno_año[2020].columns)
    
    uno_lista=[]
    for i in Uno_año[2020].columns:
        x=Uno_año[2020][i].sum()
        uno_lista.append(x)
        
        
    uno_total=sum(uno_lista)
    
    for i in Uno_año[2020].columns:
        indice=i
        lista_auxiliar_uno=[]
        for j in range(len(Uno_año[2020])):
            x=float(Uno_año[2020][indice][j]/uno_total)
            lista_auxiliar_uno.append(x)
        uno_prob[indice]=lista_auxiliar_uno
    
    uno_pred=pd.DataFrame(index=Uno_año[2020].index,columns=Uno_año[2020].columns)
    
    for i in list(Uno_año[2020].columns):
        valor=round(uno_prob[i]*pred_uno)
        uno_pred[i]=valor
        
    #AGREGo fila con nombre de curso
    idx = pd.MultiIndex.from_product([['curso'],['curso']],names=['CAMPUS', 'PROGRAMA'])
    nueva_fila=pd.DataFrame([[k]*len(uno_pred.columns)],index=idx, columns=uno_pred.columns)
    
    uno_pred_2=nueva_fila.append(uno_pred)
    uno_pred_2['PLAN']=uno_plan
    
    
    lista_pred.append(uno_pred_2)

prediccion_materias=pd.concat(lista_pred)

#exporto
prediccion_materias.to_csv('out\CURSO\prediccion_materias.csv', sep=',')



#Analisis cartera
nombre_archivo_cartera="CE_Cartera_vencida.txt"
cartera=pd.read_csv(ruta+nombre_archivo_cartera, delimiter="\t" , encoding='latin-1')
cartera_2=cartera.drop_duplicates(subset=['MATRICULA'], keep='last').reset_index(drop=True).fillna(0)

cartera_2['tiene_beca']=np.where(cartera_2['TOTAL VENCIDO']<-1,1,0)

cartera_2['convenio_economico']= np.where(cartera_2['CONVENIO ECONÓMICO']=='SI',1,0)

cartera_2['nuevo_ingreso']=np.where(cartera_2['TIPO DE ALUMNO']=='Nuevo Ingreso',1,0)

cartera_2['deuda_30_dias']=np.where(cartera_2['DIAS DE 0 A 30']!=0,1,0)
cartera_2['deuda_60_dias']=np.where(cartera_2['DIAS DE 31 A 60']!=0,1,0)
cartera_2['deuda_90_dias']=np.where(cartera_2['DIAS DE 61 A 90']!=0,1,0)
cartera_2['deuda_120_dias']=np.where(cartera_2['DIAS DE 91 A 120']!=0,1,0)
cartera_2['deuda_121_dias']=np.where(cartera_2['MAS DE 121 DIAS']!=0,1,0)


cartera_2['nivel_deuda']= np.where(cartera_2['deuda_30_dias']==1,'30 días','sin_deuda')
cartera_2['nivel_deuda']= np.where(cartera_2['deuda_60_dias']==1,'60 días',cartera_2['nivel_deuda'])
cartera_2['nivel_deuda']= np.where(cartera_2['deuda_90_dias']==1,'90 días',cartera_2['nivel_deuda'])
cartera_2['nivel_deuda']= np.where(cartera_2['deuda_120_dias']==1,'120 días',cartera_2['nivel_deuda'])
cartera_2['nivel_deuda']= np.where(cartera_2['deuda_121_dias']==1,'mas de 121 días',cartera_2['nivel_deuda'])

cartera_3=cartera_2[(cartera_2['TIPO DE ALUMNO']=='Reingreso')|(cartera_2['TIPO DE ALUMNO']=='Nuevo Ingreso')]

tabla_c=pd.merge(tabla_5,cartera_2[['MATRICULA','TOTAL VENCIDO', 'tiene_beca','nivel_deuda',
                                   'convenio_economico', 'nuevo_ingreso', 'deuda_30_dias', 'deuda_60_dias',
       'deuda_90_dias', 'deuda_120_dias', 'deuda_121_dias']], on='MATRICULA', how="inner")


tabla_modelo_c=tabla_c[['Estatus Alumno_ACTIVO','promedio', 'ultimo semestre', 'TOTAL MATERIAS CURSADAS BANNER', 'materias pagadas',
                 'terminó último periodo', 'año inicial','año final','periodo final','codigo_estatus','tiene_beca',
                   'convenio_economico', 'deuda_30_dias', 'deuda_60_dias','deuda_90_dias', 'deuda_120_dias', 'deuda_121_dias','nuevo_ingreso']]

y_cartera=tabla_modelo_c['Estatus Alumno_ACTIVO']
x_cartera=tabla_modelo_c.drop(['Estatus Alumno_ACTIVO'], axis=1)


kmeans_c=KMeans(n_clusters=5)
kmeans_c.fit(x_cartera)

y_pred_cartera=kmeans_c.predict(x_cartera)

print(silhouette_score(x_cartera,y_pred_cartera))

tabla_c['cluster']=y_pred_cartera
tabla_modelo_c['cluster']=y_pred_cartera

lista_c=list(tabla_c['cluster'].value_counts().index)


tabla_c['nombre_cluster']=np.where(tabla_c['cluster']==lista_c[4],'desertor_casi_graduado',0)
tabla_c['nombre_cluster']=np.where(tabla_c['cluster']==lista_c[1],'desertor',tabla_c['nombre_cluster'])
tabla_c['nombre_cluster']=np.where(tabla_c['cluster']==lista_c[0],'riesgo_desertar',tabla_c['nombre_cluster'])
tabla_c['nombre_cluster']=np.where(tabla_c['cluster']==lista_c[3],'constante',tabla_c['nombre_cluster'])
tabla_c['nombre_cluster']=np.where(tabla_c['cluster']==lista_c[2],'casi_graduado',tabla_c['nombre_cluster'])


tabla_clus_exportar=tabla_c[['promedio', 'ultimo semestre', 'materias pagadas', 'MATRICULA', 'PIDM',
       'ESTATUS-INSCRIPCION-ALUMNO', 'PERIODO-INICIO-CURRICULA',
       'PERIODO-ULTIMO-CURSADO', 'PERIODO-TIPOALUM-HISTORICO', 'NOMBRE ALUMNO',
       'TOTAL MATERIAS CURSADAS BANNER', 'CAMPUS', 'PROGRAMA',
       'codigo_estatus', 'estatus_alumno','TOTAL VENCIDO', 'tiene_beca',
       'nivel_deuda', 'convenio_economico', 'nuevo_ingreso','cluster','nombre_cluster',]]


#exporto
tabla_clus_exportar.to_csv('out\\alumnos_con_cartera_clusters.csv', sep=',')



#modelo estatus con la cartera
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts


tabla_modelo2_c=tabla_c[['codigo_estatus','promedio', 'ultimo semestre', 'TOTAL MATERIAS CURSADAS BANNER', 'materias pagadas',
                 'terminó último periodo', 'año inicial','año final','periodo final','tiene_beca',
                   'convenio_economico', 'deuda_30_dias', 'deuda_60_dias','deuda_90_dias', 'deuda_120_dias', 'deuda_121_dias','nuevo_ingreso']]


y2_cartera=tabla_modelo2_c['codigo_estatus']
x2_cartera=tabla_modelo2_c.drop(['codigo_estatus'], axis=1)


x2_c_train, x2_c_test, y2_c_train, y2_c_test=tts(x2_cartera,y2_cartera)

modelo_multi=LogisticRegression(multi_class='multinomial',solver='newton-cg',max_iter=3000,tol=0.01).fit(x2_c_train,y2_c_train)


y2_c_pred=modelo_multi.predict(x2_c_test)

matriz_2=pd.crosstab(y2_c_test,y2_c_pred, margins=True,)

print('la probabilidad de un falso pasivo es: ',(matriz_2[1][0]+matriz_2[2][0])/matriz_2['All'][0]*100, '%')
print('la probabilidad de un falso activo es: ',(matriz_2[0][1]+matriz_2[2][1])/matriz_2['All'][1]*100, '%')
print('la probabilidad de un falso inactivo es: ',(matriz_2[0][2]+matriz_2[1][2])/matriz_2['All'][2]*100, '%')

#guardo modelo

Pkl_Filename = "modelo_alumnos.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(modelo_multi, file)
    
    
    
    


