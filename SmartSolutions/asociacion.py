import streamlit as st
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori         # Para la implementación de reglas de asociación

@st.cache(suppress_st_warning=True)
def archivo():
    archivo = st.file_uploader("Archivo", type=['csv'])
    if archivo is not None:
        return archivo

def mainAsociacion():
    st.header("Módulo: Reglas de asociacion")
    st.write("Este programa permite generar reglas de asociación a partir de datos de ventas de una tienda.")
    datosMovies = archivo()
    if datosMovies is not None:
        df = pd.read_csv(datosMovies, header=None)#Los primeros datos nos lo toma como datos y no como un encabezado

        Transacciones = df.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida' (recomendable) o: 7460*20=149200
        ListaM = pd.DataFrame(Transacciones)
        ListaM['Frecuencia'] = 0 #Valor temporal
        #Se agrupa los elementos
        ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
        ListaM = ListaM.rename(columns={0 : 'Item'})
        if st.checkbox('Mostrar los datos cargados del archivo usado para las reglas de asociación'):
            st.subheader("A continuación, se muestran los datos cargados:")
            st.dataframe(df)
            #df.head(6) #Solo se van a mostrar 6 filas
            #Se incluyen todas las transacciones en una sola lista
            
            #Se crea una lista con las transacciones
            st.subheader("Transacciones:")
            st.dataframe(Transacciones)

            #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
            
            st.subheader("Matriz de transacciones:")
            st.dataframe(ListaM)

            #Se muestra la lista de las películas menos populares a las más populares
            st.write("Lista de las películas menos populares a las más populares:")
            st.dataframe(ListaM)

        if st.checkbox('Mostrar gráficas'):
            # Se muestra el gráfico de las películas más populares a las menos populares
            grafica = plt.figure(figsize=(20,30))
            plt.xlabel('Películas')
            plt.ylabel('Frecuencia')
            plt.barh(ListaM['Item'], ListaM['Frecuencia'],color='green')
            plt.title('Películas más populares a las menos populares')
            st.pyplot(grafica)


        #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
        #level=0 especifica desde el primer índice hasta el último
        MoviesLista = df.stack().groupby(level=0).apply(list).tolist()

        st.subheader("Ingresa los valores deseados para esta configuración del algoritmo: ")
        min_support = st.number_input("Minimo de soporte", min_value=0.0, max_value=10.0, value=0.01, step=0.01)
        min_confidence = st.number_input("Minimo de confianza", min_value=0.0, max_value=10.0, value=0.3, step=0.01)
        min_lift = st.number_input("Minimo de lift", min_value=0.0, max_value=10.0, value=2.0, step=1.0)

        if st.checkbox('Mostrar resultados'):
            
            ReglasC1 = apriori(MoviesLista, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)
            Resultado = list(ReglasC1)
            st.write("Reglas de asociación encontradas:")
            st.write(len(Resultado))

            # Mostrar las reglas de asociación
            if st.checkbox('Mostrar reglas de asociación'):
                for item in Resultado:
                    #El primer índice de la lista
                    Emparejar = item[0]
                    items = [x for x in Emparejar]
                    
                    st.write("Regla: " + str(item[0]))

                    #El segundo índice de la lista
                    st.write("Soporte: ", item[1]*100, "%")

                    #El tercer índice de la lista
                    st.write("Confianza: ", item[2][0][2]*100, "%")
                    st.write("Lift: " + str(item[2][0][3])) 
                    st.write("=====================================") 
mainAsociacion()


