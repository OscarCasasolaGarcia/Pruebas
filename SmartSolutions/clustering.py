import streamlit as st
import clusteringJ, clusteringP, clusteringJP, app
def mainClustering():
    st.header("Módulo: Clustering")
    st.sidebar.header("Selecciona el algoritmo que quieras implementar:")
    clusterOp = st.sidebar.radio("Algoritmo", ["Clustering Jerárquico (Ascendente)", "Clustering Particional (K-Means)", "Clustering Jerárquico-Particional"])
    if clusterOp == "Clustering Jerárquico (Ascendente)":
        st.title("Clustering Jerárquico (Ascendente)")
        clusteringJ.main()
    elif clusterOp == "Clustering Particional (K-Means)":
        st.title("Clustering Particional (K-Means)")
        clusteringP.main()
    elif clusterOp == "Clustering Jerárquico-Particional":
        st.title("Clustering Jerárquico-Particional")
        clusteringJP.main()
mainClustering()