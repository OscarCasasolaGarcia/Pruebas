from numpy.lib.shape_base import split
import streamlit as st
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori         # Para la implementación de reglas de asociación

st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

st.title("Módulo: Reglas de asociación")
datosAsociacion = st.file_uploader("Selecciona un archivo para trabajar las reglas de asociación", type=["csv"])
if datosAsociacion is not None:
    datosRAsociacion = pd.read_csv(datosAsociacion, header=None)

    opcionVisualizacionAsociacion = st.select_slider('Selecciona qué parte de este algoritmo quieres configurar: ', options=["Visualización", "Procesamiento","Implementación del algoritmo"])
    if opcionVisualizacionAsociacion == "Visualización":
        st.header("Visualización de los datos")
        st.dataframe(datosRAsociacion)

    if opcionVisualizacionAsociacion == "Procesamiento":
        Transacciones = datosRAsociacion.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida' (recomendable) o: 7460*20=149200
        ListaM = pd.DataFrame(Transacciones)
        ListaM['Frecuencia'] = 0 #Valor temporal
        #Se agrupa los elementos
        ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
        ListaM = ListaM.rename(columns={0 : 'Item'})
        
        column1, column2 = st.columns(2)
        #Se crea una lista con las transacciones
        column1.subheader("Transacciones:")
        column1.dataframe(Transacciones)

        #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
        
        column2.subheader("Matriz de transacciones:")
        column2.dataframe(ListaM)

        #Se muestra la lista de las películas menos populares a las más populares
        st.subheader("Elementos de los menos populares a los más populares:")
        st.dataframe(ListaM)

        with st.spinner("Generando gráfica..."):
            st.subheader("De manera gráfica: ")
            # Se muestra el gráfico de las películas más populares a las menos populares
            grafica = plt.figure(figsize=(20,30))
            plt.xlabel('Frecuencia')
            plt.ylabel('Elementos')
            plt.barh(ListaM['Item'], ListaM['Frecuencia'],color='green')
            plt.title('Elementos de los menos populares a los más populares')
            st.pyplot(grafica)

    if opcionVisualizacionAsociacion == "Implementación del algoritmo":
        #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
        #level=0 especifica desde el primer índice hasta el último
        MoviesLista = datosRAsociacion.stack().groupby(level=0).apply(list).tolist()

        st.subheader("Ingresa los valores deseados para esta configuración del algoritmo: ")
        colu1, colu2, colu3 = st.columns(3)
        min_support =  colu1.number_input("Minimo de soporte", min_value=0.0, value=0.01, step=0.01)
        min_confidence = colu2.number_input("Minimo de confianza", min_value=0.0, value=0.3, step=0.01)
        min_lift = colu3.number_input("Minimo de lift", min_value=0.0, value=2.0, step=1.0)

        if st.checkbox("Mostrar resultados: "):
            
            ReglasC1 = apriori(MoviesLista, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)
            Resultado = list(ReglasC1)
            st.success("Reglas de asociación encontradas: "+ str(len(Resultado)))

            # Mostrar las reglas de asociación
            if st.checkbox('Mostrar las reglas de asociación encontradas: '):
                if len(Resultado) == 0: 
                    st.warning("No se encontraron reglas de asociación")
                else:
                    c = st.container()
                    col1, col2, col3, col4, col5 = st.columns([1.3,2,1,1,1])
                    with st.container():
                        col1.subheader("Num. regla")
                        col2.subheader("Regla")
                        col3.subheader("Soporte")
                        col4.subheader("Confianza")
                        col5.subheader("Lift")
                        for item in Resultado:
                            with col1:
                                #El primer índice de la lista
                                st.info(str(Resultado.index(item)+1))
                                Emparejar = item[0]
                                items = [x for x in Emparejar]
                            with col2:
                                #Regla
                                st.success("("+str(", ".join(item[0]))+")")
                                
                            with col3:
                                # Soporte
                                st.success(str(round(item[1] * 100,2))+ " %")

                            with col4:
                                #Confianza
                                st.success(str(round(item[2][0][2]*100,2))+ " %")
                            
                            with col5:
                                #Lift
                                st.success(str(round(item[2][0][3],2))) 
                    
                    # Concluir las reglas de asociación
                    conclusions = st.text_area("En este espacio, se pueden anotar las conclusiones a las que se llegaron a partir de los resultados obtenidos en las reglas de asociación:", "")
                    st.subheader(conclusions)
