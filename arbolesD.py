import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
#%matplotlib inline 
import streamlit as st            # Para la generación de gráficas interactivas
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection

st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

st.title('Módulo: Árboles de decisión')
datosArboles = st.file_uploader("Seleccione un archivo para trabajar con los árboles de decisión: ", type=["csv"])
if datosArboles is not None:
    datosArbolesDecision = pd.read_csv(datosArboles)

    datosDelPronostico = []
    for i in range(0, len(datosArbolesDecision.columns)):
        datosDelPronostico.append(datosArbolesDecision.columns[i])
    
    opcionArbol1O2 = st.radio("Selecciona el tipo de árbol de decisión que deseas utilizar: ", ("Árbol de decisión (Regresión)", "Árbol de decisión (Clasificación)"))
    
    if opcionArbol1O2 == "Árbol de decisión (Regresión)":

        opcionVisualizacionArbolD = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"], value="Evaluación Visual")

        if opcionVisualizacionArbolD == "Evaluación Visual":
            st.subheader("Evaluación visual de los datos cargados: ")
            st.dataframe(datosArbolesDecision)
            st.subheader("Datos estadísticos de los datos cargados: ")
            st.dataframe(datosArbolesDecision.describe())

            st.subheader("Gráficamente")
            variablePronostico = st.selectbox("Selecciona una variable a visualizar", datosArbolesDecision.columns.drop('IDNumber'),index=4)
            if st.checkbox("Da click para cargar la gráfica (puede tardar un poco)"):
                with st.spinner('Cargando gráfica...'):
                    plt.figure(figsize=(20, 5))
                    plt.plot(datosArbolesDecision['IDNumber'], datosArbolesDecision[variablePronostico], color='green', marker='o', label=variablePronostico)
                    plt.xlabel('Paciente')
                    plt.ylabel(variablePronostico)
                    plt.title('Pacientes con tumores cancerígenos')
                    plt.grid(True)
                    plt.legend()
                    st.pyplot()
            
        if opcionVisualizacionArbolD == "Matriz de correlaciones":
            # MATRIZ DE CORRELACIONES
            MatrizCorr = datosArbolesDecision.corr(method='pearson')
            st.header("Matriz de correlaciones: ")
            st.dataframe(MatrizCorr)

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

        if opcionVisualizacionArbolD == "Aplicación del algoritmo":
            st.header('Definición de variables predictoras (X) y variable clase (Y)')
            st.subheader("Recordando la matriz de correlaciones: ")
            MatrizCorr = datosArbolesDecision.corr(method='pearson')
            st.dataframe(MatrizCorr)

            st.subheader('Variables Predictoras')
            datosADeciR = st.multiselect("Datos", datosDelPronostico)
            X = np.array(datosArbolesDecision[datosADeciR]) 
            if X.size > 0:
                with st.expander("Da click aquí para visualizar el dataframe de las variables predictoras que seleccionaste:"):
                    st.dataframe(X)

                st.subheader('Variable Clase')
                variablePronostico = st.selectbox("Variable a pronosticar", datosArbolesDecision.columns.drop(datosADeciR),index=3)
                Y = np.array(datosArbolesDecision[variablePronostico])
                with st.expander("Da click aquí para visualizar el dataframe con la variable clase que seleccionaste:"):
                    st.dataframe(Y)
        
                try:
                    # Aplicación del algoritmo: Regresión Logística
                    # Se importan las bibliotecas necesarias 
                    from sklearn import linear_model # Para la regresión lineal / pip install scikit-learn
                    from sklearn import model_selection 
                    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

                    st.header('Criterio de división')
                    testSize = st.slider('Selecciona el tamaño del test', min_value=0.2, max_value=0.3, value=0.2, step=0.01)
                    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=testSize, random_state=1234, shuffle=True)
                    # Datos de entrenamiento: 70, 75 u 80% de los datos
                    # Datos de prueba: 20, 25 o 30% de los datos

                    # DIVISIÓN DE LOS DATOS EN ENTRENAMIENTO Y PRUEBA
                    #st.dataframe(X_train)
                    #st.dataframe(Y_train)

                    # SE ENTRENA EL MODELO A TRAVÉS DE UN ÁRBOL DE DECISIÓN (REGRESIÓN)
                    # EXPLICACIÓN 
                    st.header('Parámetros del árbol de decisión: ')
                    st.markdown("""
                    * **max_depth**. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.
                    * **min_samples_leaf**. Indica la cantidad mínima de datos que debe tener un nodo hoja.
                    * **min_samples_split**. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.
                    * **criterion**. Indica la función que se utilizará para dividir los datos. Puede ser (ganancia de información) gini y entropy (Clasificación). Cuando el árbol es de regresión se usan funciones como el error cuadrado medio (MSE). """)

                    st.write("Selecciona los valores que requieras para entrenar el modelo: ")
                    choiceProfuncidad = st.select_slider('Máxima profundidad del árbol (max_depth)', options=["None","Valores numéricos"], value="None")
                    column1, column2, column3 = st.columns(3)
                    if choiceProfuncidad == "None":
                        Max_depth = None
                    elif choiceProfuncidad == "Valores numéricos":
                        Max_depth = column1.number_input('Máxima profundidad del árbol (max_depth)', min_value=1, value=8)

                    Min_samples_split = column2.number_input('min_samples_split', min_value=1, value=2)
                    Min_samples_leaf = column3.number_input('min_samples_leaf', min_value=1, value=1)
                    Criterio = st.selectbox('criterion', options=["squared_error", "friedman_mse", "absolute_error", "poisson"])
                    
                    PronosticoAD = DecisionTreeRegressor(max_depth=Max_depth, min_samples_split=Min_samples_split, min_samples_leaf=Min_samples_leaf, criterion=Criterio,random_state=0)
                    #PronosticoAD = DecisionTreeRegressor(max_depth=8, min_samples_split=4, min_samples_leaf=2)
                    
                    PronosticoAD.fit(X_train, Y_train)
                    #Se genera el pronóstico
                    Y_Pronostico = PronosticoAD.predict(X_test)
                    st.subheader('Datos del test vs Datos del pronóstico')
                    Valores = pd.DataFrame(Y_test, Y_Pronostico)
                    st.dataframe(Valores)

                    st.subheader('Gráficamente: ')
                    plt.figure(figsize=(20, 5))
                    plt.plot(Y_test, color='green', marker='o', label='Y_test')
                    plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
                    plt.xlabel('Paciente')
                    plt.ylabel('Tamaño del tumor')
                    plt.title('Pacientes con tumores cancerígenos')
                    plt.grid(True)
                    plt.legend()
                    st.pyplot()

                    # Reporte de clasificación
                    st.subheader('Reporte de clasificación')
                    with st.expander("Da click aquí para ver el reporte de clasificación"):
                        st.success('Criterio: '+str(PronosticoAD.criterion))
                        st.success('Importancia variables: '+str(PronosticoAD.feature_importances_))
                        st.success("MAE: "+str(mean_absolute_error(Y_test, Y_Pronostico)))
                        st.success("MSE: "+str(mean_squared_error(Y_test, Y_Pronostico)))
                        st.success("RMSE: "+str(mean_squared_error(Y_test, Y_Pronostico, squared=False)))   #True devuelve MSE, False devuelve RMSE
                        st.success('Score (exactitud promedio de la validación): '+str(r2_score(Y_test, Y_Pronostico).round(6)*100)+" %")
                    
                    st.subheader('Importancia de las variables')
                    Importancia = pd.DataFrame({'Variable': list(datosArbolesDecision[datosADeciR]),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                    st.table(Importancia)


                    import graphviz
                    from sklearn.tree import export_graphviz
                    # Se crea un objeto para visualizar el árbol
                    # Se incluyen los nombres de las variables para imprimirlos en el árbol
                    st.subheader('Árbol de decisión')
                    from sklearn.tree import plot_tree
                    if st.checkbox('Visualizar árbol de decisión (puede tardar un poco)'):
                        with st.spinner('Generando árbol de decisión...'):
                            plt.figure(figsize=(16,16))  
                            plot_tree(PronosticoAD, feature_names = list(datosArbolesDecision[datosADeciR]))
                            st.pyplot()

                    from sklearn.tree import export_text
                    if st.checkbox('Visualizar árbol en formato de texto: '):
                        Reporte = export_text(PronosticoAD, feature_names = list(datosArbolesDecision[datosADeciR]))
                        st.text(Reporte)

                    Elementos = export_graphviz(PronosticoAD, feature_names = list(datosArbolesDecision[datosADeciR]))  
                    Arbol = graphviz.Source(Elementos)
                    st.download_button(
                        label="Haz click aquí para descargar el árbol de decisión generado (extensión SVG)",
                        data=Arbol.pipe(format='svg'),
                        file_name="ArbolDecisionR.svg",
                        mime="image/svg"
                        )

                    st.markdown("### **El árbol generado se puede leer en el siguiente orden:** ")
                    st.markdown("""
                    1. La decisión que se toma para dividir el nodo.
                    2. El tipo de criterio que se utilizó para dividir cada nodo.
                    3. Cuántos valores tiene ese nodo.
                    4. Valores promedio.
                    5. Por último, el valor pronosticado en ese nodo. """)

                    st.subheader('Pronóstico basado en el modelo establecido')
                    with st.expander("Da click aquí para pronosticar los datos que gustes"):
                        st.subheader('Predicción de casos')
                        sujetoN = st.text_input("Ingrese el nombre o ID del paciente que se desea pronosticar: ")
                        dato = []
                        for p in range(len(datosADeciR)):
                            dato.append(st.number_input(datosADeciR[p][:], step=0.1))
                        
                        if st.checkbox("Dar pronóstico: "):
                            resultado = PronosticoAD.predict([dato])[0]
                            st.info("Con un algoritmo que tiene una exactitud promedio del: "+str(r2_score(Y_test, Y_Pronostico).round(6)*100)+"%, el pronóstico de la variable '"+str(variablePronostico)+"' fue de "+str(resultado)+" para el paciente: "+str(sujetoN)+".")
                except:
                    st.warning("Por favor, selecciona parámetros válidos para el árbol de decisión")

            elif X.size == 0:
                st.warning("No se ha seleccionado ninguna variable")

    if opcionArbol1O2 == "Árbol de decisión (Clasificación)":
        opcionVisualizacionArbolD = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"], value="Evaluación Visual")

        if opcionVisualizacionArbolD == "Evaluación Visual":
            st.dataframe(datosArbolesDecision)
            st.header("Visualización de los datos")
            variablePronostico = st.selectbox("Variable a clasificar", datosDelPronostico,index=1)
            st.write(datosArbolesDecision.groupby(variablePronostico).size())

            # Seleccionar los datos que se quieren visualizar
            try:
                datos = st.multiselect("Datos", datosDelPronostico, default=[datosDelPronostico[2], datosDelPronostico[3]])
                dato1=datos[0][:]
                dato2=datos[1][:]

                if st.checkbox("Gráfico de dispersión: "):
                    with st.spinner("Cargando gráfico de dispersión..."):
                        sns.scatterplot(x=dato1, y=dato2, data=datosArbolesDecision, hue=variablePronostico)
                        plt.title('Gráfico de dispersión')
                        plt.xlabel(dato1)
                        plt.ylabel(dato2)
                        st.pyplot() 

            except:
                st.warning("Selecciona solo dos datos")
                dato1=datosDelPronostico[0]
                dato2=datosDelPronostico[1]

            if st.checkbox("Matriz de correlaciones con el propósito de seleccionar variables significativas (puede tardar un poco): "):
                with st.spinner("Cargando matriz de correlaciones..."):
                    sns.pairplot(datosArbolesDecision, hue=variablePronostico)
                    st.pyplot()
            
        if opcionVisualizacionArbolD == "Matriz de correlaciones":
            MatrizCorr = datosArbolesDecision.corr(method='pearson')
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

        if opcionVisualizacionArbolD == "Aplicación del algoritmo":
            st.header('Definición de variables predictoras (X) y variable clase (Y)')
            MatrizCorr = datosArbolesDecision.corr(method='pearson')
            st.subheader("Recordando la matriz de correlaciones: ")
            st.dataframe(MatrizCorr)

            # SELECCIONAR VARIABLES A CLASIFICAR
            st.subheader('Selección de la variable Clase')
            st.markdown('La variable clase debe contener valores **BINARIOS**')
            variablePronostico = st.selectbox("Variable a clasificar", datosArbolesDecision.columns,index=1)

            # Comprobando que la variable clase sea binaria
            if datosArbolesDecision[variablePronostico].nunique() == 2:
                col1, col2, col3 = st.columns(3)
                # Comprobando el tipo de dato de la variable clase
                if type(datosArbolesDecision[variablePronostico].value_counts().index[1]) and type(datosArbolesDecision[variablePronostico].value_counts().index[0]) != np.int64:
                    with st.expander("Click para ver el dataframe original: "):
                        st.subheader("Dataframe original: ")
                        st.dataframe(datosArbolesDecision)
                    
                    col1.info("Selecciona las etiquetas que gustes...")
                    col2.info("Etiqueta: "+str(datosArbolesDecision[variablePronostico].value_counts().index[0]))
                    col3.info("Etiqueta: "+str(datosArbolesDecision[variablePronostico].value_counts().index[1]))
                    binario1 = col2.text_input("", datosArbolesDecision[variablePronostico].value_counts().index[0])
                    binario2 = col3.text_input("", datosArbolesDecision[variablePronostico].value_counts().index[1])

                    col2.warning("La etiqueta '"+str(datosArbolesDecision[variablePronostico].value_counts().index[0])+"', cambió por la etiqueta: "+binario1)
                    col3.warning("La etiqueta '"+str(datosArbolesDecision[variablePronostico].value_counts().index[1])+"', cambió por la etiqueta: "+binario2)

                    with st.expander("Click para ver el nuevo dataframe: "):
                        st.subheader("Dataframe corregido: ")
                        datosArbolesDecision = datosArbolesDecision.replace({str(datosArbolesDecision[variablePronostico].value_counts().index[1]): binario2, str(datosArbolesDecision[variablePronostico].value_counts().index[0]): binario1})
                        st.dataframe(datosArbolesDecision)
                        Y = np.array(datosArbolesDecision[variablePronostico])
                    
                # Variables predictoras 
                st.subheader('Variables Predictoras')
                # Seleccionar los datos que se quieren visualizar
                datos = st.multiselect("Selecciona las variables predictoras", datosArbolesDecision.columns.drop(variablePronostico))
                X = np.array(datosArbolesDecision[datos]) 
                if X.size > 0:
                    with st.expander("Da click aquí para visualizar el dataframe de las variables predictoras que seleccionaste:"):
                        st.dataframe(X)
                
                    try:
                        from sklearn.tree import DecisionTreeClassifier
                        from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
                        from sklearn import model_selection
                        # Aplicación del algoritmo: Regresión Logística
                        st.header('Criterio de división')
                        testSize = st.slider('Selecciona el tamaño del test', min_value=0.2, max_value=0.3, value=0.2, step=0.01)
                        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=testSize, random_state=0, shuffle=True)
                        # Datos de entrenamiento: 70, 75 u 80% de los datos
                        # Datos de prueba: 20, 25 o 30% de los datos

                        # DIVISIÓN DE LOS DATOS EN ENTRENAMIENTO Y PRUEBA
                        #st.dataframe(X_train)
                        #st.dataframe(Y_train)
                        # SE ENTRENA EL MODELO A TRAVÉS DE UN ÁRBOL DE DECISIÓN (REGRESIÓN)
                        # EXPLICACIÓN 
                        st.header('Parámetros del árbol de decisión: ')
                        st.markdown("""
                        * **max_depth**. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.
                        * **min_samples_leaf**. Indica la cantidad mínima de datos que debe tener un nodo hoja.
                        * **min_samples_split**. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.
                        * **criterion**. Indica la función que se utilizará para dividir los datos. Puede ser (ganancia de información) gini y entropy (Clasificación). Cuando el árbol es de regresión se usan funciones como el error cuadrado medio (MSE). """)

                        st.write("Selecciona los valores que requieras para entrenar el modelo: ")
                        choiceProfuncidad = st.select_slider('Máxima profundidad del árbol (max_depth)', options=["None","Valores numéricos"], value="None")
                        column1, column2, column3 = st.columns(3)
                        if choiceProfuncidad == "None":
                            Max_depth = None
                        elif choiceProfuncidad == "Valores numéricos":
                            Max_depth = column1.number_input('Máxima profundidad del árbol (max_depth)', min_value=1, value=8)

                        Min_samples_split = column2.number_input('min_samples_split', min_value=1, value=2)
                        Min_samples_leaf = column3.number_input('min_samples_leaf', min_value=1, value=1)
                        Criterio = st.selectbox('criterion', options=("gini", "entropy"), index=0)
                        
                        # Se entrena el modelo a partir de los datos de entrada
                        ClasificacionAD = DecisionTreeClassifier(criterion=Criterio, max_depth=Max_depth, min_samples_split=Min_samples_split, min_samples_leaf=Min_samples_leaf,random_state=0)
                        ClasificacionAD.fit(X_train, Y_train)

                        #Se etiquetan las clasificaciones
                        Y_Clasificacion = ClasificacionAD.predict(X_validation)
                        st.markdown('Se etiquetan las clasificaciones (Real vs Clasificado)')
                        Valores = pd.DataFrame(Y_validation, Y_Clasificacion)
                        st.dataframe(Valores)


                        # Se calcula la exactitud promedio de la validación
                        st.subheader('Exactitud promedio de la validación: ')
                        st.success(str(ClasificacionAD.score(X_validation, Y_validation).round(6)*100)+" %")

                        # Matriz de clasificación
                        st.subheader('Matriz de clasificación')
                        Y_Clasificacion = ClasificacionAD.predict(X_validation)
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
                            st.success("Criterio: "+str(ClasificacionAD.criterion))
                            importancia = ClasificacionAD.feature_importances_.tolist()
                            
                            st.success("Importancia de las variables: "+str(importancia))
                            st.success("Exactitud: "+ str(ClasificacionAD.score(X_validation, Y_validation)*100)+" %")
                            precision = float(classification_report(Y_validation, Y_Clasificacion).split()[10])*100
                            st.success("Precisión: "+ str(precision)+ "%")
                            st.success("Tasa de error: "+str((1-ClasificacionAD.score(X_validation, Y_validation))*100)+"%")
                            sensibilidad = float(classification_report(Y_validation, Y_Clasificacion).split()[11])*100
                            st.success("Sensibilidad: "+ str(sensibilidad)+ "%")
                            especificidad = float(classification_report(Y_validation, Y_Clasificacion).split()[6])*100
                            st.success("Especificidad: "+ str(especificidad)+"%")
                        

                        st.subheader('Importancia de las variables')
                        Importancia = pd.DataFrame({'Variable': list(datosArbolesDecision[datos]),
                                'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                        st.table(Importancia)

                    except:
                        st.warning("No se pudo realizar la clasificación porque no se ha hecho una correcta selección de variables")
                    
                    import graphviz
                    from sklearn.tree import export_graphviz
                    # Se crea un objeto para visualizar el árbol
                    # Se incluyen los nombres de las variables para imprimirlos en el árbol
                    st.subheader('Árbol de decisión (Clasificación)')

                    from sklearn.tree import plot_tree
                    if st.checkbox('Visualizar árbol de decisión (puede tardar un poco)'):
                        with st.spinner('Generando árbol de decisión...'):
                            plt.figure(figsize=(16,16))  
                            plot_tree(ClasificacionAD, feature_names = list(datosArbolesDecision[datos]),class_names=Y_Clasificacion)
                            st.pyplot()

                    from sklearn.tree import export_text
                    if st.checkbox('Visualizar árbol en formato de texto: '):
                        Reporte = export_text(ClasificacionAD, feature_names = list(datosArbolesDecision[datos]))
                        st.text(Reporte)
                    

                    #############################################
                    Elementos = export_graphviz(ClasificacionAD, feature_names = list(datosArbolesDecision[datos]), class_names=Y_Clasificacion)
                    Arbol = graphviz.Source(Elementos)

                    st.download_button(
                        label="Haz click para descargar el árbol de decisión generado (extensión SVG)",
                        data=Arbol.pipe(format='svg'),
                        file_name="ArbolDecisionC.svg",
                        mime="image/svg"
                        )
                    
                    st.markdown("### **El árbol generado se puede leer en el siguiente orden:** ")
                    st.markdown("""
                    1. La decisión que se toma para dividir el nodo.
                    2. El tipo de criterio que se usó para dividir cada nodo.
                    3. Cuantos valores tiene ese nodo.
                    4. Valores promedio.
                    5. Por último, el valor clasificado en ese nodo. """)


                    st.subheader('Clasificación de datos basado en el modelo establecido')
                    with st.expander("Da click aquí para clasificar los datos que gustes"):
                        st.subheader('Clasificación de casos')
                        sujetoN = st.text_input("Ingrese el nombre o ID del sujeto que desea clasificar: ")

                        dato = []
                        for p in range(len(datos)):
                            dato.append(st.number_input(datos[p][:], step=0.1))
                        
                        if st.checkbox("Dar clasificación: "):
                            if ClasificacionAD.predict([dato])[0] == binario2:
                                st.error("Con un algoritmo que tiene una exactitud del: "+str(round(ClasificacionAD.score(X_validation, Y_validation)*100,2))+"%, la clasificación para el paciente: "+str(sujetoN)+" fue de 0 (CERO), es decir, el diagnóstico fue "+str(binario2).upper())
                            elif ClasificacionAD.predict([dato])[0] == binario1:
                                st.success("Con un algoritmo que tiene una exactitud del: "+str(round(ClasificacionAD.score(X_validation, Y_validation)*100,2))+ "%, la clasificación para el paciente: "+str(sujetoN)+" fue de 1 (UNO), es decir, el diagnóstico fue "+str(binario1).upper())
                            else:
                                st.warning("El resultado no pudo ser determinado, intenta hacer una buena selección de variables")
                    
                elif X.size == 0:
                    st.warning("No se ha seleccionado ninguna variable")

            elif datosArbolesDecision[variablePronostico].nunique() != 2:
                st.warning("Por favor, selecciona una variable Clase (a clasificar) que contenga valores binarios...")
