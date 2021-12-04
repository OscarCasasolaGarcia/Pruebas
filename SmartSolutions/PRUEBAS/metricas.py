import streamlit as st
import pandas as pd                         # Para la manipulación y análisis de datos
import numpy as np                          # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datos
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance # Para el cálculo de distancias 

def main():
    datos = st.file_uploader("Selecciona el archivo con el que quieras trabajar con las Métricas de Distancia:", type=["csv"])
    if datos is not None:
        datosMetricas = pd.read_csv(datos) 
        if st.checkbox('Mostrar datos'):
            st.dataframe(datosMetricas)
        if st.checkbox('Matriz de distancias Euclideana'):
            # DstEuclidiana = cdist(Hipoteca, Hipoteca, metric='euclidean') # Calcula TODA la matriz de distancias 
            #Matriz de distancias de 4x4 objetos
            DstEuclidiana = cdist(datosMetricas.iloc[0:4], datosMetricas.iloc[0:4], metric='euclidean') #Distancia entre pares de objetos (4 objetos en este caso)
            matrizEuclidiana = pd.DataFrame(DstEuclidiana)
            st.write(matrizEuclidiana)
            if st.checkbox('Distancia Euclidiana entre dos objetos'):
                #Matriz euclidiana entre dos objetos  
                #Calculando la distancia entre dos objetos 
                Objeto1 = datosMetricas.iloc[0]
                Objeto2 = datosMetricas.iloc[1]
                distanciaEuclidiana = distance.euclidean(Objeto1, Objeto2)
                st.write(distanciaEuclidiana)
        if st.checkbox('Matriz de distancias Chebyshev'):
            # DstChebyshev = cdist(Hipoteca, Hipoteca, metric='chebyshev') # Calcula TODA la matriz de distancias 
            #Matriz de distancias de 4x4 objetos
            DstChebyshev = cdist(datosMetricas.iloc[0:4], datosMetricas.iloc[0:4], metric='chebyshev') #Distancia entre pares de objetos (4 objetos en este caso)
            matrizChebyshev = pd.DataFrame(DstChebyshev)
            st.write(matrizChebyshev)
            if st.checkbox('Distancia de Chebyshev entre dos objetos'):
                #Calculando la distancia de Chebyshev entre dos objetos 
                Objeto1 = datosMetricas.iloc[0]
                Objeto2 = datosMetricas.iloc[1]
                dstChebyshev = distance.chebyshev(Objeto1, Objeto2)
                st.write(dstChebyshev)
        if st.checkbox('Matriz de distancias Manhattan'):
            # DstManhattan = cdist(Hipoteca, Hipoteca, metric='cityblock') # Calcula TODA la matriz de distancias 
            #Matriz de distancias de 4x4 objetos
            DstManhattan = cdist(datosMetricas.iloc[0:4], datosMetricas.iloc[0:4], metric='cityblock') #Distancia entre pares de objetos (4 objetos en este caso)
            matrizManhattan = pd.DataFrame(DstManhattan)
            st.write(matrizManhattan)
            if st.checkbox('Distancia de Manhattan entre dos objetos'):
                #Calculando la distancia de Manhattan entre dos objetos 
                Objeto1 = datosMetricas.iloc[0]
                Objeto2 = datosMetricas.iloc[1]
                dstManhattan = distance.cityblock(Objeto1, Objeto2)
                st.write(dstManhattan)
        if st.checkbox('Matriz de distancias Minkowski'):
            # DstMinkowski = cdist(Hipoteca, Hipoteca, metric='minkowski',p=1.5) # p es el orden de la distancia 
            #Matriz de distancias de 4x4 objetos
            DstMinkowski = cdist(datosMetricas.iloc[0:4], datosMetricas.iloc[0:4], metric='minkowski', p=1.5) #Distancia entre pares de objetos (4 objetos en este caso)
            matrizMinkowski = pd.DataFrame(DstMinkowski)
            st.write(matrizMinkowski)
            if st.checkbox('Distancia de Minkowski entre dos objetos'):
                #Calculando la distancia de Minkowski entre dos objetos 
                Objeto1 = datosMetricas.iloc[0]
                Objeto2 = datosMetricas.iloc[1]
                dstMinkowski = distance.minkowski(Objeto1, Objeto2, p=1.5)
                st.write(dstMinkowski)


main()