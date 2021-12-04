import streamlit as st
import os

def main():
    st.sidebar.header("Selecciona el módulo que quieras implementar:")
    menu = ["Pantalla Principal", "Reglas de Asociación", "Métricas de distancia", "Clustering","Clasificación (R. Logística)"]
    option = st.sidebar.selectbox("Módulo", menu)

    if option == "Pantalla Principal":
        st.title("SmartsSolutions")
        imagen = st.image("https://www.muycomputerpro.com/wp-content/uploads/2021/03/inversion-inteligencia-artificial-europa-2021.jpg")
        st.write("""eervevververver""")

    if option == "Reglas de Asociación":
        import asociacion
        asociacion.mainAsociacion()

    if option == "Métricas de distancia":
        import metricas
        metricas.mainMetricas()

    if option == "Clustering":
        import clustering
        clustering.mainClustering()

    if option == "Clasificación (R. Logística)":
        import clasificacion
        clasificacion.mainClasificacion()

if __name__ == "__main__":
    main()

# Para ejeuctarlo en la terminal:
# activate IA
# streamlit run app.py
