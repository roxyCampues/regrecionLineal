import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Ruta fija del archivo CSV
RUTA_ARCHIVO_CSV = '/home/emexsis/Escritorio/regresionLineal1.0/regrecionLineal/StudentPerformanceFactors 20Data.csv'

# Función para cargar datos desde un archivo CSV
def cargar_datos_csv(ruta_archivo):
    datos = pd.read_csv(ruta_archivo)
    x = datos['Hours_Studied'].values  # Cambia según el nombre real de la columna
    y = datos['Exam_Score'].values      # Cambia según el nombre real de la columna
    return x, y

# Función para solicitar datos manualmente o cargar un CSV
def solicitar_datos():
    # En lugar de preguntar, simplemente carga el archivo CSV
    x, y = cargar_datos_csv(RUTA_ARCHIVO_CSV)
    return x, y

# Cargar datos
x, y = solicitar_datos()

# Verificar si hay al menos 10 puntos de datos
if len(x) < 10 or len(y) < 10:
    print("Error: Necesitas al menos 10 puntos de datos para realizar el análisis.")
else:
    # 1. Visualización de los datos recolectados
    plt.scatter(x, y, color='blue', label='Datos recolectados')
    plt.title('Datos de Estudio y Resultados del Examen')
    plt.xlabel('Horas Estudiadas')
    plt.ylabel('Puntuación del Examen')
    plt.legend()
    plt.show()

    # 2. Aplicar el método de mínimos cuadrados usando NumPy
    coeficientes = np.polyfit(x, y, 1)  # Ajuste lineal (grado 1)
    m, b = coeficientes  # m es la pendiente, b es la intersección (ordenada al origen)

    # Generar predicciones de y en base a la línea ajustada
    y_pred = m * x + b

    # 3. Visualización de la línea ajustada
    plt.scatter(x, y, color='blue', label='Datos originales')
    plt.plot(x, y_pred, color='red', label=f'Línea ajustada: y = {m:.2f}x + {b:.2f}')
    plt.title('Regresión Lineal')
    plt.xlabel('Horas Estudiadas')
    plt.ylabel('Puntuación del Examen')
    plt.legend()
    plt.show()

    # 4. Calcular el coeficiente de determinación R^2
    r2 = r2_score(y, y_pred)

    # Imprimir los coeficientes de la línea ajustada
    print(f'Pendiente (m): {m:.4f}')
    print(f'Intersección (b): {b:.4f}')
    print(f'Coeficiente de determinación R^2: {r2:.4f}')

    # Interpretación básica de R^2
    if r2 == 1:
        print("La línea ajustada explica perfectamente los datos.")
    elif r2 >= 0.7:
        print("El ajuste es bueno. La línea ajustada explica la mayor parte de la variabilidad.")
    elif r2 >= 0.4:
        print("El ajuste es moderado. La línea ajustada explica parte de la variabilidad, pero no toda.")
    else:
        print("El ajuste es débil. La línea ajustada no explica bien la relación entre los datos.")
