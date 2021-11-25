import streamlit as st
import pandas as pd                         # Para la manipulación y análisis de datosMetricas
import numpy as np                          # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datosMetricas
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance # Para el cálculo de distancias 
import seaborn as sns


st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib
st.title("Módulo: Metricas de distancia")
datosMetricas = st.file_uploader("Selecciona un archivo para trabajar con las Métricas de Distancia:", type=["csv"])
if datosMetricas is not None:
    datosMetricasMetricas = pd.read_csv(datosMetricas) 
    st.header("Datos subidos: ")
    st.dataframe(datosMetricasMetricas)

    opcionVisualizacionMetricas = st.select_slider('Selecciona qué métrica de distancia quieres visualizar: ', options=["Euclidiana", "Chebyshev","Manhattan","Minkowski"])
    if opcionVisualizacionMetricas == "Euclidiana":
        st.subheader("Distancia Euclidiana")
        DstEuclidiana = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='euclidean') # Calcula TODA la matriz de distancias 
        matrizEuclidiana = pd.DataFrame(DstEuclidiana)
        if st.checkbox('Matriz de distancias Euclidiana de todos los objetos'):
            with st.spinner('Cargando matriz de distancias Euclidiana...'):
                st.dataframe(matrizEuclidiana)
                st.subheader("Observando gráficamente la matriz de distancias Euclidiana: ")
                plt.figure(figsize=(10,10))
                plt.imshow(matrizEuclidiana, cmap='icefire_r')
                plt.colorbar()
                st.pyplot()
        
        if st.checkbox('Distancia Euclidiana entre dos objetos'):
            with st.spinner('Cargando distancia Euclidiana entre dos objetos...'):
                #Calculando la distancia entre dos objetos 
                st.subheader("Selecciona dos objetos para calcular la distancia entre ellos: ")
                columna1, columna2 = st.columns([1,3])
                with columna1:
                    objeto1 = st.selectbox('Objeto 1: ', options=matrizEuclidiana.columns)
                    objeto2 = st.selectbox('Objeto 2: ', options=matrizEuclidiana.columns)
                    distanciaEuclidiana = distance.euclidean(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2])
                    st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaEuclidiana))
                with columna2:
                    plt.figure(figsize=(9,9))
                    plt.grid(True)
                    plt.title("Distancia Euclidiana entre los dos objetos seleccionados")
                    plt.scatter(distanciaEuclidiana, distanciaEuclidiana, c='red',edgecolors='black')
                    plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                    plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                    plt.annotate('  '+str(distanciaEuclidiana.round(2)), xy=(distanciaEuclidiana, distanciaEuclidiana), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaEuclidiana, distanciaEuclidiana))
                    st.pyplot()

        if st.checkbox('Distancia Euclidiana entre dos objetos de tu elección'):
            with st.spinner('Cargando distancia Euclidiana entre dos objetos de tu elección...'):
                try:
                    #Calculando la distancia entre dos objetos 
                    st.subheader("Inserta las características de los objetos para calcular la distancia entre ellos: ")
                    columna1, columna2,columna3 = st.columns([1,1,1])
                    with columna1:
                        dimension = st.number_input('Selecciona el número de dimensiones que requieras: ', min_value=1, value=1)
                    
                    objeto1 = []
                    objeto2 = []
                    for p in range(0,dimension):
                        objeto1.append(columna2.number_input('Objeto 1, posición: '+str(p),value=0))
                        objeto2.append(columna3.number_input('Objeto 2, posición: '+str(p),value=0))
                        distanciaEuclidiana = distance.euclidean(objeto1, objeto2)
                        
                    st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaEuclidiana))
                    plt.figure(figsize=(9,9))
                    plt.grid(True)
                    plt.title("Distancia Euclidiana entre los dos objetos seleccionados")
                    plt.scatter(distanciaEuclidiana, distanciaEuclidiana, c='red',edgecolors='black')
                    plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                    plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                    plt.annotate('  '+str(distanciaEuclidiana.round(2)), xy=(distanciaEuclidiana, distanciaEuclidiana), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaEuclidiana, distanciaEuclidiana))
                    st.pyplot()
                except:
                    st.warning("No se han podido calcular las distancias, intenta con otros valores...")
    
    if opcionVisualizacionMetricas == "Chebyshev":
        st.subheader("Distancia Chebyshev")
        DstChebyshev = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='chebyshev') # Calcula TODA la matriz de distancias
        matrizChebyshev = pd.DataFrame(DstChebyshev)
        if st.checkbox('Matriz de distancias Chebyshev de todos los objetos'):
            with st.spinner('Cargando matriz de distancias Chebyshev...'):
                st.dataframe(matrizChebyshev)
                st.subheader("Observando gráficamente la matriz de distancias Chebyshev: ")
                plt.figure(figsize=(10,10))
                plt.imshow(matrizChebyshev, cmap='icefire_r')
                plt.colorbar()
                st.pyplot()

        if st.checkbox('Distancia Chebyshev entre dos objetos'):
            with st.spinner('Cargando distancia Chebyshev entre dos objetos...'):
                #Calculando la distancia entre dos objetos 
                st.subheader("Selecciona dos objetos para calcular la distancia entre ellos: ")
                columna1, columna2 = st.columns([1,3])
                with columna1:
                    objeto1 = st.selectbox('Objeto 1: ', options=matrizChebyshev.columns)
                    objeto2 = st.selectbox('Objeto 2: ', options=matrizChebyshev.columns)
                    distanciaChebyshev = distance.chebyshev(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2])
                    st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaChebyshev))
                with columna2:
                    plt.figure(figsize=(9,9))
                    plt.grid(True)
                    plt.title("Distancia Chebyshev entre los dos objetos seleccionados")
                    plt.scatter(distanciaChebyshev, distanciaChebyshev, c='red',edgecolors='black')
                    plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                    plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                    plt.annotate('  '+str(distanciaChebyshev.round(2)), xy=(distanciaChebyshev, distanciaChebyshev), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaChebyshev, distanciaChebyshev))
                    st.pyplot()

        if st.checkbox('Distancia Chebyshev entre dos objetos de tu elección'):
            with st.spinner('Cargando distancia Chebyshev entre dos objetos de tu elección...'):
                try:
                    #Calculando la distancia entre dos objetos 
                    st.subheader("Inserta las características de los objetos para calcular la distancia entre ellos: ")
                    columna1, columna2,columna3 = st.columns([1,1,1])
                    with columna1:
                        dimension = st.number_input('Selecciona el número de dimensiones que requieras: ', min_value=1, value=1)
                    
                    objeto1 = []
                    objeto2 = []
                    for p in range(0,dimension):
                        objeto1.append(columna2.number_input('Objeto 1, posición: '+str(p),value=0))
                        objeto2.append(columna3.number_input('Objeto 2, posición: '+str(p),value=0))
                        distanciaChebyshev = distance.chebyshev(objeto1, objeto2)
                        
                    st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaChebyshev))
                    
                    plt.figure(figsize=(9,9))
                    plt.grid(True)
                    plt.title("Distancia Chebyshev entre los dos objetos seleccionados")
                    plt.scatter(distanciaChebyshev, distanciaChebyshev, c='red',edgecolors='black')
                    plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                    plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                    plt.annotate('  '+str(distanciaChebyshev.round(2)), xy=(distanciaChebyshev, distanciaChebyshev), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaChebyshev, distanciaChebyshev))
                    st.pyplot()
                except:
                    st.warning("No se han podido calcular las distancias, intenta con otros valores...")

    if opcionVisualizacionMetricas == "Manhattan":
        st.subheader("Distancia Manhattan")
        DstManhattan = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='cityblock') # Calcula TODA la matriz de distancias
        matrizManhattan = pd.DataFrame(DstManhattan)
        if st.checkbox('Matriz de distancias Manhattan de todos los objetos'):
            with st.spinner('Cargando matriz de distancias Manhattan...'):
                st.dataframe(matrizManhattan)
                st.subheader("Observando gráficamente la matriz de distancias Manhattan: ")
                plt.figure(figsize=(10,10))
                plt.imshow(matrizManhattan, cmap='icefire_r')
                plt.colorbar()
                st.pyplot()

        if st.checkbox('Distancia Manhattan entre dos objetos'):
            with st.spinner('Cargando distancia Manhattan entre dos objetos...'):
                #Calculando la distancia entre dos objetos 
                st.subheader("Selecciona dos objetos para calcular la distancia entre ellos: ")
                columna1, columna2 = st.columns([1,3])
                with columna1:
                    objeto1 = st.selectbox('Objeto 1: ', options=matrizManhattan.columns)
                    objeto2 = st.selectbox('Objeto 2: ', options=matrizManhattan.columns)
                    distanciaManhattan = distance.cityblock(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2])
                    st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaManhattan))
                with columna2:
                    plt.figure(figsize=(9,9))
                    plt.grid(True)
                    plt.title("Distancia Manhattan entre los dos objetos seleccionados")
                    plt.scatter(distanciaManhattan, distanciaManhattan, c='red',edgecolors='black')
                    plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                    plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                    plt.annotate('  '+str(distanciaManhattan.round(2)), xy=(distanciaManhattan, distanciaManhattan), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaManhattan, distanciaManhattan))
                    st.pyplot()

        if st.checkbox('Distancia Manhattan entre dos objetos de tu elección'):
            with st.spinner('Cargando distancia Manhattan entre dos objetos de tu elección...'):
                try:
                    #Calculando la distancia entre dos objetos 
                    st.subheader("Inserta las características de los objetos para calcular la distancia entre ellos: ")
                    columna1, columna2,columna3 = st.columns([1,1,1])
                    with columna1:
                        dimension = st.number_input('Selecciona el número de dimensiones que requieras: ', min_value=1, value=1)
                    
                    objeto1 = []
                    objeto2 = []
                    for p in range(0,dimension):
                        objeto1.append(columna2.number_input('Objeto 1, posición: '+str(p),value=0))
                        objeto2.append(columna3.number_input('Objeto 2, posición: '+str(p),value=0))
                        distanciaManhattan = distance.cityblock(objeto1, objeto2)
                        
                    st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaManhattan))
                    
                    plt.figure(figsize=(9,9))
                    plt.grid(True)
                    plt.title("Distancia Manhattan entre los dos objetos seleccionados")
                    plt.scatter(distanciaManhattan, distanciaManhattan, c='red',edgecolors='black')
                    plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                    plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                    plt.annotate('  '+str(distanciaManhattan.round(2)), xy=(distanciaManhattan, distanciaManhattan), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaManhattan, distanciaManhattan))
                    st.pyplot()
                except:
                    st.warning("No se han podido calcular las distancias, intenta con otros valores...")

    if opcionVisualizacionMetricas == "Minkowski":
        st.subheader("Distancia Minkowski")
        DstMinkowski = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='minkowski',p=1.5) # Calcula TODA la matriz de distancias
        matrizMinkowski = pd.DataFrame(DstMinkowski)
        if st.checkbox('Matriz de distancias Minkowski de todos los objetos'):
            with st.spinner('Cargando matriz de distancias Minkowski...'):
                st.dataframe(matrizMinkowski)
                st.subheader("Observando gráficamente la matriz de distancias Minkowski: ")
                plt.figure(figsize=(10,10))
                plt.imshow(matrizMinkowski, cmap='icefire_r')
                plt.colorbar()
                st.pyplot()

        if st.checkbox('Distancia Minkowski entre dos objetos'):
            with st.spinner('Cargando distancia Minkowski entre dos objetos...'):
                #Calculando la distancia entre dos objetos 
                st.subheader("Selecciona dos objetos para calcular la distancia entre ellos: ")
                columna1, columna2 = st.columns([1,3])
                with columna1:
                    objeto1 = st.selectbox('Objeto 1: ', options=matrizMinkowski.columns)
                    objeto2 = st.selectbox('Objeto 2: ', options=matrizMinkowski.columns)
                    distanciaMinkowski = distance.minkowski(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2], p=1.5)
                    st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaMinkowski))
                with columna2:
                    plt.figure(figsize=(9,9))
                    plt.grid(True)
                    plt.title("Distancia Minkowski entre los dos objetos seleccionados")
                    plt.scatter(distanciaMinkowski, distanciaMinkowski, c='red',edgecolors='black')
                    plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                    plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                    plt.annotate('  '+str(distanciaMinkowski.round(2)), xy=(distanciaMinkowski, distanciaMinkowski), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaMinkowski, distanciaMinkowski))
                    st.pyplot()

        if st.checkbox('Distancia Minkowski entre dos objetos de tu elección'):
            with st.spinner('Cargando distancia Minkowski entre dos objetos de tu elección...'):
                try:
                    #Calculando la distancia entre dos objetos 
                    st.subheader("Inserta las características de los objetos para calcular la distancia entre ellos: ")
                    columna1, columna2,columna3 = st.columns([1,1,1])
                    with columna1:
                        dimension = st.number_input('Selecciona el número de dimensiones que requieras: ', min_value=1, value=1)

                    objeto1 = []
                    objeto2 = []
                    for p in range(0,dimension):
                        objeto1.append(columna2.number_input('Objeto 1, posición: '+str(p),value=0))
                        objeto2.append(columna3.number_input('Objeto 2, posición: '+str(p),value=0))
                        distanciaMinkowski = distance.minkowski(objeto1, objeto2, p=1.5)
                        
                    st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaMinkowski))

                    plt.figure(figsize=(9,9))
                    plt.grid(True)
                    plt.title("Distancia Minkowski entre los dos objetos seleccionados")
                    plt.scatter(distanciaMinkowski, distanciaMinkowski, c='red',edgecolors='black')
                    plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                    plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                    plt.annotate('  '+str(distanciaMinkowski.round(2)), xy=(distanciaMinkowski, distanciaMinkowski), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaMinkowski, distanciaMinkowski))
                    st.pyplot()
                except:
                    st.warning("No se han podido calcular las distancias, intenta con otros valores...")
