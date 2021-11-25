import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
#%matplotlib inline 
import streamlit as st            # Para la generación de gráficas interactivas
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Para escalar los datos
#Se importan las bibliotecas de clustering particional
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

st.title('Clustering particional')
datosClusterP = st.file_uploader("Seleccione un archivo para trabajar con el Clustering Particional", type=["csv"])
if datosClusterP is not None:
    datosClusteringP = pd.read_csv(datosClusterP)
    #Hipoteca.info()

    datosDelPronostico = []
    for i in range(0, len(datosClusteringP.columns)):
        datosDelPronostico.append(datosClusteringP.columns[i])

    
    opcionVisualizacionClustersP = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Selección de variables + algoritmo"])

    if opcionVisualizacionClustersP == "Evaluación Visual":
        st.title("EVALUACIÓN VISUAL")
        st.header("Evaluación visual de los datos cargados: ")
        st.dataframe(datosClusteringP.head())
        st.markdown("**Selecciona la variable a pronosticar:** ")
        variablePronostico = st.selectbox("", datosClusteringP.columns,index=9)
        st.write(datosClusteringP.groupby(variablePronostico).size())
        try:
            # Seleccionar los datos que se quieren visualizar
            st.markdown("**Selecciona dos variables que quieras visualizar en el gráfico de dispersión:** ")
            datos = st.multiselect("", datosClusteringP.columns, default=[datosClusteringP.columns[4],datosClusteringP.columns[0]])
            dato1=datos[0][:]
            dato2=datos[1][:]
        except:
            st.warning("Selecciona solo dos datos...")
            dato1=datosDelPronostico[0]
            dato2=datosDelPronostico[1]

        with st.spinner("Cargando datos..."):
            if st.checkbox("Gráfico de dispersión"):
                sns.scatterplot(x=dato1, y=dato2, data=datosClusteringP, hue=variablePronostico)
                plt.title('Gráfico de dispersión')
                plt.xlabel(dato1)
                plt.ylabel(dato2)
                st.pyplot()
            with st.spinner("Cargando datos..."):
                if st.checkbox('Ver la matriz de correlaciones con el propósito de seleccionar variables significativas:'):
                    sns.pairplot(datosClusteringP, hue=variablePronostico)
                    st.pyplot()
    
    if opcionVisualizacionClustersP == "Matriz de correlaciones":
        st.title("MATRIZ DE CORRELACIONES")
        # MATRIZ DE CORRELACIONES
        MatrizCorr = datosClusteringP.corr(method='pearson')
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
    
    if opcionVisualizacionClustersP == "Selección de variables + algoritmo":
        try: 
            st.header("Selecciona las variables para trabajar: ")
            variableSeleccionadas = st.multiselect("", datosClusteringP.columns, default=[datosClusteringP.columns[0],datosClusteringP.columns[1]])
            MatrizClusteringP = np.array(datosClusteringP[variableSeleccionadas])
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
                
                datosClusteringP = datosClusteringP[variableSeleccionadas]
                datosClusteringP['clusterP'] = MParticional.labels_
                st.dataframe(datosClusteringP)

                #Cantidad de elementos en los clusters
                numClusters = datosClusteringP.groupby(['clusterP'])['clusterP'].count() 
                st.subheader("Cantidad de elementos en los clusters: ")
                st.table(numClusters)

                # Centroides de los clusters
                CentroidesP = datosClusteringP.groupby(['clusterP'])[variableSeleccionadas].mean()
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