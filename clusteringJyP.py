from matplotlib import text
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
#%matplotlib inline 
import streamlit as st            # Para la generación de gráficas interactivas
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Para escalar los datos

#Librerías para el clustering jerárquico 
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#Librerías para Clustering Particional
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

st.title('Módulo: Clustering')
datosCluster = st.file_uploader("Seleccione un archivo para trabajar con clustering", type=["csv"])
if datosCluster is not None:
    datosClustering = pd.read_csv(datosCluster)

    datosDelPronostico = []
    for i in range(0, len(datosClustering.columns)):
        datosDelPronostico.append(datosClustering.columns[i])

    opcionClustering1O2 = st.radio("Selecciona el algoritmo de clustering que deseas implementar: ", ('Clustering Jerárquico (Ascendente)', 'Clustering Particional (K-Means)'))

    if opcionClustering1O2 == "Clustering Jerárquico (Ascendente)":
        opcionVisualizacionClustersJ = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"])

        if opcionVisualizacionClustersJ == "Evaluación Visual":
            st.header("EVALUACIÓN VISUAL")
            st.subheader("Datos cargados: ")
            st.dataframe(datosClustering)
            st.subheader("Selecciona la variable a pronosticar: ")
            variablePronostico = st.selectbox("", datosClustering.columns,index=9)
            st.write(datosClustering.groupby(variablePronostico).size())
            try:
                # Seleccionar los datos que se quieren visualizar
                st.subheader("Selecciona dos variables que quieras visualizar en el gráfico de dispersión: ")
                datos = st.multiselect("", datosClustering.columns, default=[datosClustering.columns[4],datosClustering.columns[0]])
                dato1=datos[0][:]
                dato2=datos[1][:]

                with st.spinner("Cargando datos..."):
                    if st.checkbox("Gráfico de dispersión"):
                        sns.scatterplot(x=dato1, y=dato2, data=datosClustering, hue=variablePronostico)
                        plt.title('Gráfico de dispersión')
                        plt.xlabel(dato1)
                        plt.ylabel(dato2)
                        st.pyplot()

                with st.spinner("Cargando datos..."):
                    if st.checkbox('Ver el gráfico de dispersión de todas las variables con el propósito de seleccionar variables significativas: (puede tardar un poco)'):
                        sns.pairplot(datosClustering, hue=variablePronostico)
                        st.pyplot()
            except:
                st.warning("Selecciona solo dos variables...")
                

        if opcionVisualizacionClustersJ == "Matriz de correlaciones":
            st.header("MATRIZ DE CORRELACIONES")
            # MATRIZ DE CORRELACIONES
            MatrizCorr = datosClustering.corr(method='pearson')
            st.dataframe(MatrizCorr)
            try:
                st.subheader("Selecciona una variable para observar cómo se correlaciona con las demás: ")
                variableCorrelacion = st.selectbox("", datosClustering.columns) 
                st.markdown("**Matriz de correlaciones con la variable seleccionada:** ")
                st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores 
            except:
                st.warning("Selecciona una variable con datos válidos.")

            # Mapa de calor de la relación que existe entre variables
            with st.spinner("Cargando mapa de calor..."):
                st.header("Mapa de calor de la relación que existe entre variables: ")
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(MatrizCorr)
                sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                plt.title('Mapa de calor de la correlación que existe entre variables')
                st.pyplot()
        
        if opcionVisualizacionClustersJ == "Aplicación del algoritmo":
            st.header("SELECCIÓN DE VARIABLES Y APLICACIÓN DEL ALGORITMO SELECCIONADO")
            st.subheader("Recordando la matriz de correlaciones: ")
            MatrizCorr = datosClustering.corr(method='pearson')
            st.dataframe(MatrizCorr)

            st.subheader('Selección variables para el análisis: ')
            SeleccionVariablesJ = st.multiselect("Selecciona las variables para hacer el análisis: ", datosClustering.columns)
            MatrizClusteringJ = np.array(datosClustering[SeleccionVariablesJ])
            if MatrizClusteringJ.size > 0:
                with st.expander("Da click aquí para visualizar el dataframe de las variables que seleccionaste:"):
                    st.dataframe(MatrizClusteringJ)
                # Aplicación del algoritmo: 
                estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
                MEstandarizada = estandarizar.fit_transform(MatrizClusteringJ)   # Se calculan la media y desviación y se escalan los datos
                st.subheader("MATRIZ ESTANDARIZADA: ")
                st.dataframe(MEstandarizada) 

                st.subheader("Selecciona la métrica de distancias a utilizar: ")
                metricaElegida = st.selectbox("", ('euclidean','chebyshev','cityblock','minkowski'),index=0)
                ClusterJerarquico = shc.linkage(MEstandarizada, method='complete', metric=metricaElegida)
                with st.spinner("Cargando gráfico..."):
                    graficaClusteringJ = plt.figure(figsize=(10, 5))
                    plt.title("Clustering Jerárquico (Ascendente)")
                    plt.xlabel('Observaciones')
                    plt.ylabel('Distancia')
                    Arbol = shc.dendrogram(ClusterJerarquico) #Utilizamos la matriz estandarizada
                    SelectAltura = st.slider('Selecciona a qué nivel quieres "cortar" el árbol: ', min_value=0.0, max_value=np.max(Arbol['dcoord']),step=0.1)
                    plt.axhline(y=SelectAltura, color='black', linestyle='--') # Hace un corte en las ramas
                    st.pyplot(graficaClusteringJ)
                
                numClusters = fcluster(ClusterJerarquico, t=SelectAltura, criterion='distance')
                NumClusters = len(np.unique(numClusters))
                st.success("El número de clusters elegido fue de: "+ str(NumClusters))
                
                with st.spinner("Cargando..."):
                    if st.checkbox("Ver los clusters obtenidos: "):
                        try:
                            #Se crean las etiquetas de los elementos en los clústeres
                            MJerarquico = AgglomerativeClustering(n_clusters=NumClusters, linkage='complete', affinity=metricaElegida)
                            MJerarquico.fit_predict(MEstandarizada)
                            #MJerarquico.labels_

                            datosClustering = datosClustering[SeleccionVariablesJ]
                            datosClustering['clusterH'] = MJerarquico.labels_
                            st.subheader("Dataframe con la etiqueta del cluster obtenida: ")
                            st.dataframe(datosClustering)

                            #Cantidad de elementos en los clusters
                            cantidadElementos = datosClustering.groupby(['clusterH'])['clusterH'].count() 
                            st.header("Cantidad de elementos en los clusters: ")
                            st.write(cantidadElementos)

                            # Centroides de los clusters
                            CentroidesH = datosClustering.groupby('clusterH').mean()
                            st.header("Centroides de los clusters: ")
                            st.table(CentroidesH)

                            # Interpretación de los clusters
                            st.header("Interpretación de los clusters obtenidos: ")
                            with st.expander("Haz click para visualizar los datos contenidos en cada cluster: "):
                                for i in range(NumClusters):
                                    st.subheader("Cluster "+str(i))
                                    st.write(datosClustering[datosClustering['clusterH'] == i])
                            
                            st.subheader("Interpretación de los centroides de los clusters obtenidos: ")
                            with st.expander("Haz click para visualizar los centroides obtenidos en cada cluster: "):
                                for i in range(NumClusters):
                                    st.subheader("Cluster "+str(i))
                                    st.table(CentroidesH.iloc[i])

                            with st.expander("Haz click para visualizar las conclusiones obtenidas de los centroides de cada cluster: "):
                                for n in range(NumClusters):
                                    st.subheader("Cluster "+str(n))
                                    st.markdown("**Conformado por: "+str(cantidadElementos[n])+" elementos**")
                                    for m in range(CentroidesH.columns.size):
                                        st.markdown("* Con "+str(CentroidesH.columns[m])+" promedio de: "+"**"+str(CentroidesH.iloc[n,m].round(5))+"**.")
                                
                            try: 
                                # Gráfico de barras de la cantidad de elementos en los clusters
                                st.header("Representación gráfica de los clusteres: ")
                                plt.figure(figsize=(10, 5))
                                plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
                                plt.grid()
                                st.pyplot()
                            except:
                                st.warning("No se pudo graficar.")
                        except: 
                            st.warning("No se pudo realizar el proceso de clustering, selecciona un 'corte' al árbol que sea correcto")

            elif MatrizClusteringJ.size == 0:
                st.warning("No se ha seleccionado ninguna variable.")


    if opcionClustering1O2 == "Clustering Particional (K-Means)":
        st.title('Clustering particional')

        opcionVisualizacionClustersP = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"])

        if opcionVisualizacionClustersP == "Evaluación Visual":
            st.header("EVALUACIÓN VISUAL")
            st.header("Evaluación visual de los datos cargados: ")
            st.dataframe(datosClustering)
            st.markdown("**Selecciona la variable a pronosticar:** ")
            variablePronostico = st.selectbox("", datosClustering.columns,index=9)
            st.write(datosClustering.groupby(variablePronostico).size())
            try:
                # Seleccionar los datos que se quieren visualizar
                st.markdown("**Selecciona dos variables que quieras visualizar en el gráfico de dispersión:** ")
                datos = st.multiselect("", datosClustering.columns, default=[datosClustering.columns[4],datosClustering.columns[0]])
                dato1=datos[0][:]
                dato2=datos[1][:]
            except:
                st.warning("Selecciona solo dos datos...")
                dato1=datosDelPronostico[0]
                dato2=datosDelPronostico[1]

            with st.spinner("Cargando datos..."):
                if st.checkbox("Gráfico de dispersión"):
                    sns.scatterplot(x=dato1, y=dato2, data=datosClustering, hue=variablePronostico)
                    plt.title('Gráfico de dispersión')
                    plt.xlabel(dato1)
                    plt.ylabel(dato2)
                    st.pyplot()
                with st.spinner("Cargando datos..."):
                    if st.checkbox('Ver la matriz de correlaciones con el propósito de seleccionar variables significativas:'):
                        sns.pairplot(datosClustering, hue=variablePronostico)
                        st.pyplot()
        
        if opcionVisualizacionClustersP == "Matriz de correlaciones":
            st.title("MATRIZ DE CORRELACIONES")
            # MATRIZ DE CORRELACIONES
            MatrizCorr = datosClustering.corr(method='pearson')
            st.header("Matriz de correlaciones: ")
            st.dataframe(MatrizCorr)
            try:
                st.subheader("Selecciona una variable para observar cómo se correlaciona con las demás: ")
                variableCorrelacion = st.selectbox("", MatrizCorr.columns) 
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
        
        if opcionVisualizacionClustersP == "Aplicación del algoritmo":
            try: 
                st.header("Selecciona las variables para trabajar: ")
                variableSeleccionadas = st.multiselect("", datosClustering.columns, default=[datosClustering.columns[0],datosClustering.columns[1]])
                MatrizClusteringP = np.array(datosClustering[variableSeleccionadas])
                st.dataframe(MatrizClusteringP)

                st.header('Aplicación del algoritmo: K-Means')
                
                # Aplicación del algoritmo: 
                estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
                MEstandarizada = estandarizar.fit_transform(MatrizClusteringP)   # Se calculan la media y desviación y se escalan los datos
                st.subheader("MATRIZ ESTANDARIZADA: ")
                st.dataframe(MEstandarizada) 

                try: 
                    #Definición de k clusters para K-means
                    #Se utiliza random_state para inicializar el generador interno de números aleatorios
                    k = st.slider('Selecciona el número de clusteres a implementar: ', min_value=2, max_value=20, value=12, step=1)
                    SSE = []
                    for i in range(2, k):
                        km = KMeans(n_clusters=i, random_state=0)
                        km.fit(MEstandarizada)
                        SSE.append(km.inertia_)
                    
                    #Se grafica SSE en función de k
                    plt.figure(figsize=(10, 7))
                    plt.plot(range(2, k), SSE, marker='o')
                    plt.xlabel('Cantidad de clusters *k*')
                    plt.ylabel('SSE')
                    plt.title('Elbow Method')
                    st.pyplot()

                    kl = KneeLocator(range(2, k), SSE, curve="convex", direction="decreasing")
                    st.subheader('El codo se encuentra en el cluster número: '+str(kl.elbow))

                    plt.style.use('ggplot')
                    kl.plot_knee()
                    st.pyplot()

                    #Se crean las etiquetas de los elementos en los clústeres
                    MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
                    MParticional.predict(MEstandarizada)
                    
                    datosClustering = datosClustering[variableSeleccionadas]
                    datosClustering['clusterP'] = MParticional.labels_
                    st.dataframe(datosClustering)

                    #Cantidad de elementos en los clusters
                    numClusters = datosClustering.groupby(['clusterP'])['clusterP'].count() 
                    st.subheader("Cantidad de elementos en los clusters: ")
                    st.table(numClusters)

                    # Centroides de los clusters
                    CentroidesP = datosClustering.groupby(['clusterP'])[variableSeleccionadas].mean()
                    st.subheader("Centroides de los clusters: ")
                    st.table(CentroidesP)

                    # Interpretación de los clusters
                    st.subheader("Interpretación de los clusters: ")
                    st.markdown("**ESCRIBE UNA INTERPRETACIÓN DE LOS CLUSTERES OBTENIDOS...**")
                    for p in range(0,len(numClusters)):
                        st.text_area("Interpretación del cluster: "+str(p),"Este cluster está conformado por "+str(numClusters[p])+" elementos...")

                except:
                    st.warning("Selecciona un número válido de clusteres")
                
                try:
                    st.header("Representación gráfica de los clusteres: ")
                    # Gráfica de los elementos y los centros de los clusters
                    from mpl_toolkits.mplot3d import Axes3D
                    plt.rcParams['figure.figsize'] = (10, 7)
                    plt.style.use('ggplot')
                    colores=['red', 'blue', 'green', 'yellow']
                    asignar=[]
                    for row in MParticional.labels_:
                        asignar.append(colores[row])

                    fig = plt.figure()
                    ax = Axes3D(fig)
                    ax.scatter(MEstandarizada[:, 0], 
                            MEstandarizada[:, 1], 
                            MEstandarizada[:, 2], marker='o', c=asignar, s=60)
                    ax.scatter(MParticional.cluster_centers_[:, 0], 
                            MParticional.cluster_centers_[:, 1], 
                            MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
                    st.pyplot()

                except:
                    st.warning("No se pudo graficar")

            except:
                st.warning("Selecciona variables con datos válidos...")
