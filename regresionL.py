from typing import BinaryIO
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
#%matplotlib inline 
import streamlit as st            # Para la generación de gráficas interactivas
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Para escalar los datos

st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

st.title('Módulo: Regresión Logística')
datosRegresionL = st.file_uploader("Seleccione un archivo para trabajar con la regresión logística: ", type=["csv"])
if datosRegresionL is not None:
    DatosRegresionL = pd.read_csv(datosRegresionL)

    datosDelPronostico = []
    for i in range(0, len(DatosRegresionL.columns)):
        datosDelPronostico.append(DatosRegresionL.columns[i])
    
    opcionVisualizacionRegresionL = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"])

    if opcionVisualizacionRegresionL == "Evaluación Visual":
        st.dataframe(DatosRegresionL)
        st.header("Visualización de los datos")
        variablePronostico = st.selectbox("Variable a clasificar", datosDelPronostico,index=1)
        st.write(DatosRegresionL.groupby(variablePronostico).size())

        # Seleccionar los datos que se quieren visualizar
        try:
            st.subheader("Gráfico de dispersión")
            datos = st.multiselect("Selecciona dos variables", datosDelPronostico, default=[datosDelPronostico[0], datosDelPronostico[1]])
            dato1=datos[0][:]
            dato2=datos[1][:]

            if st.checkbox("Visualizar gráfico de dispersión: "):
                with st.spinner("Cargando gráfico de dispersión..."):
                    sns.scatterplot(x=dato1, y=dato2, data=DatosRegresionL, hue=variablePronostico)
                    plt.title('Gráfico de dispersión')
                    plt.xlabel(dato1)
                    plt.ylabel(dato2)
                    st.pyplot() 

        except:
            st.warning("Selecciona solo dos datos")
            dato1=datosDelPronostico[0]
            dato2=datosDelPronostico[1]

        if st.checkbox("Visualizar gráfico de dispersión de todas las variables con el fin de seleccionar variables significativas: (puede tardar un poco)"):
            with st.spinner("Cargando matriz de correlaciones..."):
                sns.pairplot(DatosRegresionL, hue=variablePronostico)
                st.pyplot()
        
    if opcionVisualizacionRegresionL == "Matriz de correlaciones":
        # MATRIZ DE CORRELACIONES
        MatrizCorr = DatosRegresionL.corr(method='pearson')
        st.header("Matriz de correlaciones: ")
        st.dataframe(MatrizCorr)

        # SELECCIONAR VARIABLES PARA PRONOSTICAR
        try:
            st.subheader("Correlación de variables: ")
            variableCorrelacion = st.selectbox("", MatrizCorr.columns) 
            st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores
        except:
            st.warning("Selecciona una variable con datos válidos...")

        # Mapa de calor de la relación que existe entre variables
        plt.figure(figsize=(14,7))
        MatrizInf = np.triu(MatrizCorr)
        sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
        plt.title('Mapa de calor de la relación que existe entre variables')
        st.pyplot()

    if opcionVisualizacionRegresionL == "Aplicación del algoritmo":
        st.header('Definición de variables predictoras (X) y variable clase (Y)')
        MatrizCorr = DatosRegresionL.corr(method='pearson')
        st.subheader("Recordando la matriz de correlaciones: ")
        st.dataframe(MatrizCorr)

        st.subheader('Selección de la variable Clase')
        st.markdown('La variable clase debe contener valores **BINARIOS**')
        variablePronostico = st.selectbox("Variable a clasificar", DatosRegresionL.columns)

        # Comprobando que la variable clase sea binaria
        if DatosRegresionL[variablePronostico].nunique() == 2:
            col1, col2, col3 = st.columns(3)
            # Comprobando el tipo de dato de la variable clase
            if type(DatosRegresionL[variablePronostico].value_counts().index[1]) and type(DatosRegresionL[variablePronostico].value_counts().index[0]) != np.int64:
                col1.warning("Para hacer una correcta clasificación, se necesita que los datos a clasificar sean de tipo BINARIO (0,1)...")
                col2.error("La etiqueta '"+str(DatosRegresionL[variablePronostico].value_counts().index[1])+"', cambió por el valor 0")
                col3.success("La etiqueta '"+str(DatosRegresionL[variablePronostico].value_counts().index[0])+"', cambió por el valor 1")
                with st.expander("Click para ver el dataframe original: "):
                    st.subheader("Dataframe original: ")
                    st.dataframe(DatosRegresionL)
                with st.expander("Click para ver el dataframe corregido: "):
                    st.subheader("Dataframe corregido: ")
                    DatosRegresionL = DatosRegresionL.replace({str(DatosRegresionL[variablePronostico].value_counts().index[1]): 0, str(DatosRegresionL[variablePronostico].value_counts().index[0]): 1})
                    st.dataframe(DatosRegresionL)
                    Y = np.array(DatosRegresionL[variablePronostico])
            
            Y = np.array(DatosRegresionL[variablePronostico])
            # Variables predictoras
            st.subheader('Selección de las variables Predictoras')
            datos = st.multiselect("Selecciona las variables predictoras", DatosRegresionL.columns.drop(variablePronostico))
            X = np.array(DatosRegresionL[datos])
            if X.size > 0:
                with st.expander("Da click aquí para visualizar el dataframe de las variables predictoras que seleccionaste:"):
                    st.dataframe(X)
        
                # Seleccionar los datos que se quieren visualizar
                st.subheader('Visualización de datos: Variables predictoras y su correlación con la variable a clasificar')
                try:
                    datosPronostico = st.multiselect("Selecciona dos variables: ", datos)
                    datoPronostico1=datosPronostico[0][:]
                    datoPronostico2=datosPronostico[1][:]
                    
                    plt.figure(figsize=(10,7))
                    plt.scatter(X[:,datos.index(datosPronostico[0])], X[:,datos.index(datosPronostico[1])], c=DatosRegresionL[variablePronostico])
                    plt.grid()
                    plt.xlabel(datoPronostico1)
                    plt.ylabel(datoPronostico2)
                    st.pyplot()
                except:
                    st.warning("Por favor, selecciona mínimo dos variables que sean válidas para su visualización")
                    datoPronostico1=datosDelPronostico[0][:]
                    datoPronostico2=datosDelPronostico[1][:]
                
                
                try:
                    # Aplicación del algoritmo: Regresión Logística
                    # Se importan las bibliotecas necesarias 
                    from sklearn import linear_model # Para la regresión lineal / pip install scikit-learn
                    from sklearn import model_selection 
                    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

                    st.header('Criterio de división')
                    testSize = st.slider('Selecciona el tamaño del test', min_value=0.2, max_value=0.3, value=0.2, step=0.01)
                    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=testSize, random_state=1234, shuffle=True)
                    # Datos de entrenamiento: 70, 75 u 80% de los datos
                    # Datos de prueba: 20, 25 o 30% de los datos

                    # DIVISIÓN DE LOS DATOS EN ENTRENAMIENTO Y PRUEBA
                    #st.dataframe(X_train)
                    #st.dataframe(Y_train)

                    # Se entrena el modelo a partir de los datos de entrada
                    Clasificacion = linear_model.LogisticRegression() # Se crea el modelo
                    Clasificacion.fit(X_train, Y_train) # Se entrena el modelo

                    contenedorPredicciones1, contenedorPredicciones2, contenedorPredicciones3 = st.columns(3)
                    with contenedorPredicciones1:
                        # Predicciones probabilísticas
                        st.markdown('Predicciones probabilísticas de los datos de entrenamiento')
                        Probabilidad = Clasificacion.predict_proba(X_train)
                        st.dataframe(Probabilidad)

                    with contenedorPredicciones2:
                        st.markdown('Predicciones probabilísticas de los datos de validación')
                        # Predicciones probabilísticas de los datos de prueba
                        Probabilidad = Clasificacion.predict_proba(X_validation)
                        st.dataframe(Probabilidad) # A partir de las probabilidades se hacen el etiqueta de si es 1 o 0

                    with contenedorPredicciones3:
                        # Predicciones con clasificación final
                        st.markdown('Predicciones con clasificación final')
                        Predicciones = Clasificacion.predict(X_validation)
                        st.dataframe(Predicciones) # A partir de las probabilidades obtenidas anteriormente se hacen las predicciones

                    # Se calcula la exactitud promedio de la validación
                    st.subheader('Exactitud promedio de la validación: ')
                    st.success(str(Clasificacion.score(X_validation, Y_validation).round(6)*100)+" %")

                    # Matriz de clasificación
                    st.subheader('Matriz de clasificación')
                    Y_Clasificacion = Clasificacion.predict(X_validation)
                    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación'])
                    st.table(Matriz_Clasificacion)
                    
                    col1, col2 = st.columns(2)
                    col1.info('Verdaderos Positivos (VP): '+str(Matriz_Clasificacion.iloc[1,1]))
                    col2.info('Falsos Negativos (FN): '+str(Matriz_Clasificacion.iloc[1,0]))
                    col2.info('Verdaderos Negativos (VN): '+str(Matriz_Clasificacion.iloc[0,0]))
                    col1.info('Falsos Positivos (FP): '+str(Matriz_Clasificacion.iloc[0,1]))

                    # Reporte de clasificación
                    st.subheader('Reporte de clasificación')
                    with st.expander("Da click aquí para ver el reporte de clasificación"):
                        st.write(classification_report(Y_validation, Y_Clasificacion))
                        st.success("Exactitud: "+ str(Clasificacion.score(X_validation, Y_validation)*100)+" %")
                        precision = float(classification_report(Y_validation, Y_Clasificacion).split()[10])*100
                        st.success("Precisión: "+ str(precision)+ " %")
                        st.success("Tasa de error: "+str((1-Clasificacion.score(X_validation, Y_validation))*100)+" %")
                        sensibilidad = float(classification_report(Y_validation, Y_Clasificacion).split()[11])*100
                        st.success("Sensibilidad: "+ str(sensibilidad)+ " %")
                        especificidad = float(classification_report(Y_validation, Y_Clasificacion).split()[6])*100
                        st.success("Especificidad: "+ str(especificidad)+" %")
                    
                    st.subheader('Modelo de clasificación: ')
                    # Ecuación del modelo
                    st.latex(r"p=\frac{1}{1+e^{-(a+bX)}}")
                    
                    with st.expander("Da click aquí para ver la ecuación del modelo"):
                        st.success("Intercept: "+str(Clasificacion.intercept_[0]))
                        st.write("Coeficientes:\n",Clasificacion.coef_)
                        st.latex("a+bX="+str(Clasificacion.intercept_[0]))
                        for i in range(len(datos)):
                            datos[i] = datos[i].replace("_", "")
                            st.latex("+"+str(Clasificacion.coef_[0][i].round(6))+"("+str(datos[i])+")")

                    st.subheader('Clasificación basada en el modelo establecido')
                    with st.expander("Da click aquí para clasificar los datos que gustes"):
                        st.subheader('Clasificación de casos')
                        sujetoN = st.text_input("Ingrese el nombre o ID del sujeto que desea clasificar: ")

                        dato = []
                        for p in range(len(datos)):
                            dato.append(st.number_input(datos[p][:], step=0.1))
                        
                        if st.checkbox("Dar clasificación: "):
                            if Clasificacion.predict([dato])[0] == 0:
                                st.error("Con un algoritmo que tiene una exactitud del: "+str(round(Clasificacion.score(X_validation, Y_validation)*100,2))+"%, la clasificación para el sujeto "+str(sujetoN)+", tomando en cuenta como variable predictora: '"+str(variablePronostico)+"', fue de 0 (CERO)")
                            elif Clasificacion.predict([dato])[0] == 1:
                                st.success("Con un algoritmo que tiene una exactitud del: "+str(round(Clasificacion.score(X_validation, Y_validation)*100,2))+"%, la clasificación para el sujeto "+str(sujetoN)+", tomando en cuenta como variable predictora: '"+str(variablePronostico)+"', fue de 1 (UNO)")
                            else:
                                st.warning("El resultado no pudo ser determinado, intenta hacer una buena selección de variables")
                except:
                    st.warning("No se pudo realizar la clasificación porque no se ha hecho una correcta selección de variables")

            elif X.size == 0:
                st.warning("No se han seleccionado variables predictoras...")

        elif DatosRegresionL[variablePronostico].nunique() != 2:
            st.warning("La variable clase no contiene datos binarios, por lo que no se puede realizar la clasificación... intenta con otra variable")
