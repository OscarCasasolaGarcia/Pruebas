import pandas as pd               # Para la manipulación y análisis de datos / pip install pandas
import numpy as np                # Para crear vectores y matrices n dimensionales / pip install numpy
import matplotlib.pyplot as plt  # Para la generación de gráficas a partir de los datos / pip install matplotlib
import seaborn as sns             # Para la visualización de datos basado en matplotlib / pip install seaborn
import streamlit as st

def mainClasificacion():
    st.header("Módulo: Clasificación con Regresión Logística")
    st.write("Clasificación con regresión Logística	")
    datosClasificacion = st.file_uploader("Selecciona el archivo para trabajar con la clasificación", type=["csv"])
    if datosClasificacion is not None:
        dc = pd.read_csv(datosClasificacion, header=None)
        if st.checkbox('Mostrar los datos cargados del archivo usado para la clasificación: '):
            st.write("Mostrando los pacientes de acuerdo con su diagnóstico: ")
            
            # De una población total de 569 muestras, 357 presentan un tipo de cáncer benigno y 212 maligno. 
        
        menuClasificacion = ["Selecciona una opción","Normal", "Experto"]
        optionClasificacion = st.selectbox("¿Eres un usuario experto o usuario normal?: ", menuClasificacion)

        if optionClasificacion == "Selecciona una opción":
            pass
        
        if optionClasificacion == "Normal":
            st.subheader("NORMAL")
        
        if optionClasificacion == "Experto":
            st.subheader("EXPERTO")
            st.write("Gráfica de la relación de diagnosis con las demás: ")
            


mainClasificacion()
