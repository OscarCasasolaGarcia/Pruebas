import streamlit as st

st.title("SmartsSolutions")
st.sidebar.header("Selecciona el algoritmo que requieras implementar:")
algorithm = st.sidebar.radio("Algoritmo", ["Inicio", "Reglas de Asociación", "Métricas de distancia", "Clustering Jerarquico", "Clustering Particional", "Tumores", "Salir"])

if algorithm == "Inicio":
    imagen = st.image("https://www.muycomputerpro.com/wp-content/uploads/2021/03/inversion-inteligencia-artificial-europa-2021.jpg")

elif algorithm == "Reglas de Asociación":
    import asociacion
    st.title("Reglas de Asociación")
    datosMovies = st.file_uploader("Selecciona el archivo para trabajar con las Reglas de Asociación", type=["csv"])
    if datosMovies is not None:
        st.write("Archivo seleccionado: ", datosMovies)
        asociacion.main()

elif algorithm == "Métricas de distancia":
    import metricas
    st.title("Métricas de distancia")
    metricas.main()

elif algorithm == "Salir":
    st.title("Gracias por utilizar nuestros servicios")


# Para ejeuctarlo en la terminal:
# activate Deeplearning
# streamlit run app.py
