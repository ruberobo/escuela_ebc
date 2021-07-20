# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:01:19 2021

@author: GOTSA
"""

import pandas as pd
import numpy as np
import re
import seaborn as sns
import math
import statistics as stat
import matplotlib as plt
import pickle


#variables
promedio=0
semestre=0
materias_baner=0
inscripciones=0
termino=0
año_inicial=0
año_final=0
periodo_final=0
beca=0
convenio=0
dias_30=0
dias_60=0
dias_90=0
dias_120=0
dias_121=0
nuevo_ingreso=0



Pkl_Filename = "modelo_alumnos.pkl"  

#Load the Model back from file

with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)
    

from tkinter import *

from tkinter.ttk import *

window = Tk()

window.title("Alumnos ebc")

#resultado
etiqueta=Label()

#variables para circulos
selected_periodo = IntVar()
selected_beca = IntVar()
selected_convenio = IntVar()
selected_30 = IntVar()
selected_60 = IntVar()
selected_90 = IntVar()
selected_120 = IntVar()
selected_121 = IntVar()

#circulos
rad1 = Radiobutton(window,text='Sí', value=1, variable=selected_periodo)
rad2 = Radiobutton(window,text='No', value=0, variable=selected_periodo)

beca_radsi=Radiobutton(window,text='Sí', value=1, variable=selected_beca)
beca_radno=Radiobutton(window,text='No', value=0, variable=selected_beca)

convenio_radsi=Radiobutton(window,text='Sí', value=1, variable=selected_convenio)
convenio_radno=Radiobutton(window,text='No', value=0, variable=selected_convenio)

circulo_30si=Radiobutton(window,text='Sí', value=1, variable=selected_30)
circulo_30no=Radiobutton(window,text='No', value=0, variable=selected_30)

circulo_60si=Radiobutton(window,text='Sí', value=1, variable=selected_60)
circulo_60no=Radiobutton(window,text='No', value=0, variable=selected_60)

circulo_90si=Radiobutton(window,text='Sí', value=1, variable=selected_90)
circulo_90no=Radiobutton(window,text='No', value=0, variable=selected_90)

circulo_120si=Radiobutton(window,text='Sí', value=1, variable=selected_120)
circulo_120no=Radiobutton(window,text='No', value=0, variable=selected_120)

circulo_121si=Radiobutton(window,text='Sí', value=1, variable=selected_121)
circulo_121no=Radiobutton(window,text='No', value=0, variable=selected_121)

#accesorios
promedio_texto= Entry (window,font=('Calibri 15'))
semestre_spin = Spinbox(window, from_=1, to=8, width=5)
baner_spin = Spinbox(window, from_=1, to=42, width=5)
inscripciones_spin = Spinbox(window, from_=1, to=100, width=5)
años_i_spin = Spinbox(window, from_=2010, to=2025, width=5)
años_f_spin = Spinbox(window, from_=2010, to=2025, width=5)
periodo_texto= Entry (window,font=('Calibri 15'))


#funciones
def termino_periodo():
    global termino
    termino = selected_periodo.get()
    print(termino)
   
def texto_periodo():
    global periodo
    periodo=float(periodo_texto.get())
    print(periodo)   
   
def texto_promedio():
    global promedio
    promedio=float(promedio_texto.get())
    print(promedio)

def texto_semestre():
    global semestre
    semestre= float(semestre_spin.get())
    print(semestre)
    
def texto_baner():
    global materias_baner
    materias_baner= float(baner_spin.get())
    print(materias_baner)
    
def texto_inscripciones():
    global inscripciones
    inscripciones=float(inscripciones_spin.get())
    print(inscripciones)
   
    
def texto_año_inicial():
    global año_inicial
    año_inicial= float(años_i_spin.get())
    print(año_inicial)


def texto_año_final():
    global año_final
    año_final= float(años_f_spin.get())
    print(año_final)
    
def tiene_beca():
    global beca
    beca = selected_beca.get()
    print(beca)
    
    
def tiene_convenio():
    global convenio
    convenio = selected_convenio.get()
    print(convenio)
    
def tiene_30():
    global dias_30
    dias_30 = selected_30.get()
    print(dias_30)
    
def tiene_60():
    global dias_60
    dias_60 = selected_60.get()
    print(dias_60)    
    
def tiene_90():
    global dias_90
    dias_90 = selected_90.get()
    print(dias_90)
    
def tiene_120():
    global dias_120
    dias_120 = selected_120.get()
    print(dias_120)
    
def tiene_121():
    global dias_121
    dias_121 = selected_121.get()
    print(dias_121)
    
def regresion():
    lista=[promedio,semestre,materias_baner,inscripciones,termino,año_inicial,año_final,periodo_final,beca,convenio,dias_30,dias_60,
      dias_90,dias_120,dias_121,nuevo_ingreso]


    entrada=np.array([lista])

    prediccion=Pickled_LR_Model.predict(entrada)

    if prediccion==0:
        resultado='alumno pasivo'
    elif prediccion==1:
        resultado='alumno activo'
    elif prediccion==2:
        resultado='alumno inactivo'
        
    etiqueta['text']=resultado
    

#botones
btn1 = Button(window, text="¿qué promedio tiene el alumno?", command=texto_promedio)
btn2 = Button(window, text="¿semestre del alumno?", command=texto_semestre)
btn3 = Button(window, text="¿Materias baner?", command=texto_baner)
btn4 = Button(window, text="¿cuántas inscripciones?", command=texto_inscripciones)
btn5 = Button(window, text="¿terminó su último periodo?", command=termino_periodo)
btn6 = Button(window, text="¿cuál es su año inicial?", command=texto_año_inicial)
btn7 = Button(window, text="¿cuál es su año final?", command=texto_año_final)
btn8 = Button(window, text="¿cuál es su último periodo?", command=texto_periodo)

btn9 = Button(window, text="¿tiene beca?", command=tiene_beca)
btn10 = Button(window, text="¿tiene convenio económico?", command=tiene_convenio)
btn11 = Button(window, text="¿tiene deuda a 30 días?", command=tiene_30)
btn12 = Button(window, text="¿tiene deuda a 60 días?", command=tiene_60)
btn13 = Button(window, text="¿tiene deuda a 90 días?", command=tiene_90)
btn14 = Button(window, text="¿tiene deuda a 120 días?", command=tiene_120)
btn15 = Button(window, text="¿tiene deuda a mas de 121 días?", command=tiene_121)

btn16 = Button(window, text="¿qué estatus tendría el alumno?", command=regresion)

#ubicando
promedio_texto.grid(row=0, column=1)
rad1.grid(column=1, row=4)
rad2.grid(column=2, row=4)

beca_radsi.grid(column=1, row=8)
beca_radno.grid(column=2, row=8)

convenio_radsi.grid(column=1, row=9)
convenio_radno.grid(column=2, row=9)

circulo_30si.grid(column=1, row=10)
circulo_30no.grid(column=2, row=10)

circulo_60si.grid(column=1, row=11)
circulo_60no.grid(column=2, row=11)

circulo_90si.grid(column=1, row=12)
circulo_90no.grid(column=2, row=12)

circulo_120si.grid(column=1, row=13)
circulo_120no.grid(column=2, row=13)

circulo_121si.grid(column=1, row=14)
circulo_121no.grid(column=2, row=14)


semestre_spin.grid(column=1, row=1)
baner_spin.grid(column=1, row=2)
inscripciones_spin.grid(column=1, row=3)
años_i_spin.grid(column=1, row=5)
años_f_spin.grid(column=1, row=6)
periodo_texto.grid(column=1, row=7)

etiqueta.grid(column=1, row=16)


btn1.grid(column=0, row=0)
btn2.grid(column=0, row=1)
btn3.grid(column=0, row=2)
btn4.grid(column=0, row=3)
btn5.grid(column=0, row=4)
btn6.grid(column=0, row=5)
btn7.grid(column=0, row=6)
btn8.grid(column=0, row=7)
btn9.grid(column=0, row=8)
btn10.grid(column=0, row=9)
btn11.grid(column=0, row=10)
btn12.grid(column=0, row=11)
btn13.grid(column=0, row=12)
btn14.grid(column=0, row=13)
btn15.grid(column=0, row=14)
btn16.grid(column=0, row=16)



window.mainloop()


#prediccion

#lista=[promedio,semestre,materias_baner,inscripciones,termino,año_inicial,año_final,periodo_final,beca,convenio,dias_30,dias_60,
      #dias_90,dias_120,dias_121,nuevo_ingreso]


#entrada=np.array([lista])

#prediccion=Pickled_LR_Model.predict(entrada)


#if prediccion==0:
    #print('alumno pasivo')
#elif prediccion==1:
    #print('alumno activo')
#elif prediccion==2:
    #print('alumno inactivo')