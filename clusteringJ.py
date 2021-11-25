import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
#%matplotlib inline 
import streamlit as st            # Para la generación de gráficas interactivas
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Para escalar los datos
#Se importan las bibliotecas de clustering jerárquico para crear el árbol
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

st.title('Clustering Jerárquico')
datosClusterJ = st.file_uploader("Seleccione el archivo para trabajar con el Clustering Jerárquico", type=["csv"])
if datosClusterJ is not None:
    datosClusteringJ = pd.read_csv(datosClusterJ)
    #Hipoteca.info()

    datosDelPronostico = []
    for i in range(0, len(datosClusteringJ.columns)):
        datosDelPronostico.append(datosClusteringJ.columns[i])

    opcionVisualizacionClustersJ = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Selección de variables + algoritmo"])

    if opcionVisualizacionClustersJ == "Evaluación Visual":
        st.title("EVALUACIÓN VISUAL")
        st.header("Evaluación visual de los datos cargados: ")
        st.dataframe(datosClusteringJ.head())
        st.markdown("**Selecciona la variable a pronosticar:** ")
        variablePronostico = st.selectbox("", datosClusteringJ.columns,index=9)
        st.write(datosClusteringJ.groupby(variablePronostico).size())
        try:
            # Seleccionar los datos que se quieren visualizar
            st.markdown("**Selecciona dos variables que quieras visualizar en el gráfico de dispersión:** ")
            datos = st.multiselect("", datosClusteringJ.columns, default=[datosClusteringJ.columns[4],datosClusteringJ.columns[0]])
            dato1=datos[0][:]
            dato2=datos[1][:]
        except:
            st.warning("Selecciona solo dos datos...")
            dato1=datosDelPronostico[0]
            dato2=datosDelPronostico[1]

        with st.spinner("Cargando datos..."):
            if st.checkbox("Gráfico de dispersión"):
                sns.scatterplot(x=dato1, y=dato2, data=datosClusteringJ, hue=variablePronostico)
                plt.title('Gráfico de dispersión')
                plt.xlabel(dato1)
                plt.ylabel(dato2)
                st.pyplot()
            with st.spinner("Cargando datos..."):
                if st.checkbox('Ver la matriz de correlaciones con el propósito de seleccionar variables significativas:'):
                    sns.pairplot(datosClusteringJ, hue=variablePronostico)
                    st.pyplot()
    
    if opcionVisualizacionClustersJ == "Matriz de correlaciones":
        st.title("MATRIZ DE CORRELACIONES")
        # MATRIZ DE CORRELACIONES
        MatrizCorr = datosClusteringJ.corr(method='pearson')
        st.header("Matriz de correlaciones: ")
        st.dataframe(MatrizCorr)
        try:
            st.subheader("Selecciona una variable para observar cómo se correlaciona con las demás: ")
            variableCorrelacion = st.selectbox("", datosClusteringJ.columns) 
            st.subheader("Matriz de correlaciones con la variable seleccionada: ")
            st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores 
        except:
            st.warning("Selecciona una variable con datos válidos.")

        # Mapa de calor de la relación que existe entre variables
        with st.spinner("Cargando mapa de calor..."):
            st.header("Mapa de calor de la relación que existe entre variables: ")
            plt.figure(figsize=(14,7))
            MatrizInf = np.triu(MatrizCorr)
            sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
            plt.title('Mapa de calor de la relación que existe entre variables')
            st.pyplot()
    
    if opcionVisualizacionClustersJ == "Selección de variables + algoritmo":
        try:
            st.title("SELECCIÓN DE VARIABLES Y APLICACIÓN DEL ALGORITMO SELECCIONADO")
            st.header("Selecciona la variable que quieras suprimir: ")
            variableSuprimir = st.selectbox("", datosClusteringJ.columns, index=9)
            MatrizClusteringJ = np.array(datosClusteringJ[datosClusteringJ.columns.drop(variableSuprimir)])
            st.dataframe(MatrizClusteringJ)

            # Aplicación del algoritmo: 
            estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
            MEstandarizada = estandarizar.fit_transform(MatrizClusteringJ)   # Se calculan la media y desviación y se escalan los datos
            st.subheader("MATRIZ ESTANDARIZADA: ")
            st.dataframe(MEstandarizada) 

            try:
                st.subheader("Selecciona la métrica de distancias a utilizar: ")
                metricaElegida = st.selectbox("", ('euclidean', 'cityblock', 'chebyshev'))
                with st.spinner("Cargando datos..."):
                    graficaClusteringJ = plt.figure(figsize=(10, 5))
                    plt.title("Casos de hipoteca")
                    plt.xlabel('Hipoteca')
                    plt.ylabel('Distancia')
                    Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric=metricaElegida)) #Utilizamos la matriz estandarizada
                    #plt.axhline(y=6, color='orange', linestyle='--') # Hace un corte en las ramas
                    #Probar con otras mediciones de distancia (chebyshev, cityblock, etc.)
                    st.pyplot(graficaClusteringJ)
            except:
                st.warning("Selecciona una métrica válida...")
        except:
            st.warning("Selecciona una variable con datos válidos...")

        try: 
            with st.spinner("Cargando..."):
                st.subheader("Selecciona el número de clusters que tiene el dendrograma presentado anteriormente: ")
                numClusters = st.number_input("Número de clusters", min_value=1, max_value=50, value=7,step=1)
                #Se crean las etiquetas de los elementos en los clústeres
                MJerarquico = AgglomerativeClustering(n_clusters=numClusters, linkage='complete', affinity=metricaElegida)
                MJerarquico.fit_predict(MEstandarizada)
                #MJerarquico.labels_

                st.subheader("Selecciona la variable que quieras suprimir: ")
                variableSuprimir = st.selectbox("Variable a eliminar", datosClusteringJ.columns, index=9)
                datosClusteringJ = datosClusteringJ.drop(columns=[variableSuprimir])
                datosClusteringJ['clusterH'] = MJerarquico.labels_
                st.dataframe(datosClusteringJ)

                
                #Cantidad de elementos en los clusters
                cantidadElementos = datosClusteringJ.groupby(['clusterH'])['clusterH'].count() 
                st.header("Cantidad de elementos en los clusters: ")
                st.write(cantidadElementos)

                # Centroides de los clusters
                CentroidesH = datosClusteringJ.groupby('clusterH').mean()
                st.header("Centroides de los clusters: ")
                st.table(CentroidesH)

                # Interpretación de los clusters
                st.header("Interpretación de los clusters: ")
                for p in range(0,numClusters):
                    st.text_area("Interpretación del cluster: "+str(p),"Este cluster está conformado por "+str(cantidadElementos[p])+" elementos...")
                
                # Gráfico de barras de la cantidad de elementos en los clusters
                st.header("Representación gráfica de los clusteres: ")
                plt.figure(figsize=(10, 5))
                plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
                plt.grid()
                st.pyplot()
        except: 
            st.write("No se pudo realizar el proceso de clustering, selecciona las variables adecuadas.")
