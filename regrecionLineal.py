import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Ruta fija del archivo CSV
RUTA_ARCHIVO_CSV = 'StudentPerformanceFactors 20Data.csv'

# Función para cargar datos desde un archivo CSV
def cargar_datos_csv(ruta_archivo):
    datos = pd.read_csv(ruta_archivo)
    x = datos['Hours_Studied'].values  # Cambia según el nombre real de la columna
    y = datos['Exam_Score'].values      # Cambia según el nombre real de la columna
    return x, y

# Cargar datos
x, y = cargar_datos_csv(RUTA_ARCHIVO_CSV)

# Verificar si hay al menos 10 puntos de datos
if len(x) < 10 or len(y) < 10:
    print("Error: Necesitas al menos 10 puntos de datos para realizar el análisis.")
else:
    # 1. Mínimos Cuadrados: Ajuste lineal usando NumPy
    coeficientes = np.polyfit(x, y, 1)  # Ajuste lineal (grado 1)
    m, b = coeficientes  # m es la pendiente, b es la intersección (ordenada al origen)

    # Generar predicciones de y en base a la línea ajustada
    y_pred = m * x + b

    # 2. Coeficiente de Determinación R^2
    r2 = r2_score(y, y_pred)

    # Imprimir los coeficientes de la línea ajustada y R^2
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

    # 3. Visualización de los datos recolectados y la línea ajustada
    plt.scatter(x, y, color='blue', label='Datos originales')
    plt.plot(x, y_pred, color='red', label=f'Línea ajustada: y = {m:.2f}x + {b:.2f}')
    plt.title('Regresión Lineal')
    plt.xlabel('Horas Estudiadas')
    plt.ylabel('Puntuación del Examen')
    plt.legend()

    # Mostrar la gráfica
    try:
        plt.show()  # Intenta mostrar la gráfica si el entorno lo permite
    except:
        print("No se pudo mostrar la gráfica directamente, se guardará como archivo.")

    # Guardar la gráfica como archivo si no se muestra
    plt.savefig('grafica.png')
    print("La gráfica ha sido guardada como 'grafica.png'. Puedes revisarla en el explorador de archivos.")
