# Actividad 3

###########################################################################################
# --- VARIABLES ---

#variable de texto
saludo="hola a todos, soy Lady Elizabeth"
print(saludo)

#lista de numeros
num_mascotas=[1,0,2,1,2,0]
print(num_mascotas)

#diccionario: tipo de objeto que permite dar un valor 
calificaciones={"Macroeconomía" : "16,5", "Teoria de Juegos":"15,5"}
print(calificaciones)

#numerico
# VECTORES
enteros=[10]*5
print(enteros)

flotantes=[3.14]*5
print(flotantes)

complejos = [(1 + 2j)] * 5
print(complejos)

# Diccionario que contenga los vectores
diccionario_vectores={"entero": enteros, "flotante":flotantes, "complejo": complejos}
print(diccionario_vectores)

#Cadenas
frase_cadena='Dicen las estrellas que los fugases smos nosotros :)'
materias=["Microeconomía", "Teoria monetaria"]
print(frase_cadena)
print(materias)

#Lógicos o booleanos
booleanos = [True, False]

###########################################################################################
#DataFrame
import pandas as pd

# Crear un DataFrame con los datos de rendimiento en juegos
datos = {
    'Nombre': ['Lily', 'Mishell', 'Cecy', 'Karen'],
    'Edad': [28, 27, 56, 22],
    'Altura': [150,165,155, 160],
    'Telefono': [987643560, 981234789, 912398009, 988776644]
}

df = pd.DataFrame(datos)

# Mostrar el DataFrame
print(df)

###########################################################################################
# --- Importación de un archivo .xlsx ---

#librerias
import pandas as pd

#importar datos
imp_sri= pd.read_excel("ventas_SRI.xlsx")
print(imp_sri)

#estructura de los datos
imp_sri.dtypes

#nombre de las variables
imp_sri.columns

#numero e filas y columnas
imp_sri.shape

imp_sri.describe()

#imprimir las 10 primeras observaciones
imp_sri.head(10)