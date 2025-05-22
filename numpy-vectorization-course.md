# 🚀 Curso Completo: Vectorización con NumPy (De 0 a 100)

## 📋 Índice
1. [Introducción: ¿Qué es la vectorización?](#introducción)
2. [¿Por qué es importante?](#por-qué-importante)
3. [Conceptos fundamentales](#conceptos-fundamentales)
4. [De bucles a vectores](#de-bucles-a-vectores)
5. [Broadcasting: La magia de NumPy](#broadcasting)
6. [Técnicas avanzadas](#técnicas-avanzadas)
7. [Casos prácticos del mundo real](#casos-prácticos)
8. [Errores comunes y cómo evitarlos](#errores-comunes)
9. [Proyecto final integrador](#proyecto-final)

---

## 1. Introducción: ¿Qué es la vectorización? {#introducción}

### 🎯 Definición simple
La vectorización es **realizar operaciones sobre arrays completos en lugar de elemento por elemento**.

### 📊 Ejemplo visual
```python
# SIN vectorización (lento) 🐌
resultado = []
for i in range(len(lista)):
    resultado.append(lista[i] * 2)

# CON vectorización (rápido) ⚡
import numpy as np
array = np.array(lista)
resultado = array * 2
```

### 🧠 La metáfora de la fábrica
Imagina que tienes que pintar 1000 sillas:
- **Sin vectorización**: Pintas cada silla completamente antes de pasar a la siguiente
- **Con vectorización**: Aplicas una capa de pintura a TODAS las sillas al mismo tiempo

---

## 2. ¿Por qué es importante? {#por-qué-importante}

### ⏱️ Diferencias de velocidad reales
```python
import numpy as np
import time

# Crear datos de prueba
n = 1_000_000
lista_python = list(range(n))
array_numpy = np.arange(n)

# Método tradicional con bucles
start = time.time()
resultado_bucle = []
for x in lista_python:
    resultado_bucle.append(x ** 2)
tiempo_bucle = time.time() - start

# Método vectorizado
start = time.time()
resultado_vector = array_numpy ** 2
tiempo_vector = time.time() - start

print(f"Tiempo con bucle: {tiempo_bucle:.4f} segundos")
print(f"Tiempo vectorizado: {tiempo_vector:.4f} segundos")
print(f"¡Vectorización es {tiempo_bucle/tiempo_vector:.1f}x más rápida!")
```

**Resultado típico:**
```
Tiempo con bucle: 0.2500 segundos
Tiempo vectorizado: 0.0025 segundos
¡Vectorización es 100.0x más rápida!
```

### 💡 Ventajas clave
1. **Velocidad**: 10-100x más rápido
2. **Código más limpio**: Menos líneas, más legible
3. **Menos errores**: Sin índices que gestionar
4. **Uso eficiente de memoria**: NumPy optimiza el almacenamiento

---

## 3. Conceptos fundamentales {#conceptos-fundamentales}

### 📦 Arrays vs Listas
```python
# Lista de Python: elementos pueden ser de cualquier tipo
lista = [1, "hola", 3.14, [1,2,3]]

# Array de NumPy: todos los elementos del mismo tipo
array = np.array([1, 2, 3, 4])  # todos enteros
```

### 🔢 Tipos de datos (dtype)
```python
# NumPy elige el tipo automáticamente
arr_int = np.array([1, 2, 3])          # dtype: int64
arr_float = np.array([1.0, 2.0, 3.0])  # dtype: float64

# O puedes especificarlo
arr_32 = np.array([1, 2, 3], dtype=np.float32)
```

### 📐 Formas y dimensiones
```python
# Vector (1D)
vector = np.array([1, 2, 3, 4])
print(f"Forma: {vector.shape}")  # (4,)

# Matriz (2D)
matriz = np.array([[1, 2], 
                   [3, 4], 
                   [5, 6]])
print(f"Forma: {matriz.shape}")  # (3, 2)

# Tensor (3D)
tensor = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
print(f"Forma: {tensor.shape}")  # (2, 2, 2)
```

---

## 4. De bucles a vectores {#de-bucles-a-vectores}

### 🔄 Transformación paso a paso

#### Ejemplo 1: Suma de elementos
```python
# ❌ Con bucles
def suma_bucle(lista1, lista2):
    resultado = []
    for i in range(len(lista1)):
        resultado.append(lista1[i] + lista2[i])
    return resultado

# ✅ Vectorizado
def suma_vector(arr1, arr2):
    return arr1 + arr2

# Comparación
a = np.random.rand(10000)
b = np.random.rand(10000)

%timeit suma_bucle(a.tolist(), b.tolist())  # ~2.5 ms
%timeit suma_vector(a, b)                    # ~10 μs (250x más rápido!)
```

#### Ejemplo 2: Aplicar función matemática
```python
# ❌ Con bucles
def aplicar_formula_bucle(lista):
    resultado = []
    for x in lista:
        # Fórmula: f(x) = x² + 2x + 1
        resultado.append(x**2 + 2*x + 1)
    return resultado

# ✅ Vectorizado
def aplicar_formula_vector(arr):
    return arr**2 + 2*arr + 1

# La versión vectorizada es más legible Y más rápida
```

#### Ejemplo 3: Operaciones condicionales
```python
# ❌ Con bucles
def filtrar_positivos_bucle(lista):
    resultado = []
    for x in lista:
        if x > 0:
            resultado.append(x)
        else:
            resultado.append(0)
    return resultado

# ✅ Vectorizado
def filtrar_positivos_vector(arr):
    return np.where(arr > 0, arr, 0)

# O incluso más simple:
def filtrar_positivos_vector2(arr):
    return arr * (arr > 0)
```

---

## 5. Broadcasting: La magia de NumPy {#broadcasting}

### 🎨 ¿Qué es broadcasting?
Broadcasting permite operar arrays de diferentes formas sin copiar datos.

### 📏 Reglas del broadcasting
1. Los arrays se alinean por la derecha
2. Las dimensiones deben ser iguales o una debe ser 1
3. Las dimensiones faltantes se consideran 1

### 🔍 Ejemplos visuales

#### Broadcasting simple
```python
# Array + escalar
arr = np.array([1, 2, 3, 4])
resultado = arr + 10  # [11, 12, 13, 14]

# Lo que NumPy hace internamente:
# [1, 2, 3, 4] + 10
# [1, 2, 3, 4] + [10, 10, 10, 10]  # Broadcasting!
```

#### Broadcasting 2D
```python
# Matriz + vector (por filas)
matriz = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

vector_fila = np.array([10, 20, 30])
resultado = matriz + vector_fila
# [[11, 22, 33],
#  [14, 25, 36],
#  [17, 28, 39]]

# Matriz + vector (por columnas)
vector_columna = np.array([[100], 
                           [200], 
                           [300]])
resultado2 = matriz + vector_columna
# [[101, 102, 103],
#  [204, 205, 206],
#  [307, 308, 309]]
```

#### Ejemplo práctico: Normalización
```python
# Normalizar cada columna de una matriz
datos = np.random.rand(100, 3) * 100

# Media y desviación de cada columna
media = datos.mean(axis=0)      # shape: (3,)
desviacion = datos.std(axis=0)  # shape: (3,)

# Normalización con broadcasting
datos_normalizados = (datos - media) / desviacion
# Broadcasting: (100,3) - (3,) → (100,3)
```

---

## 6. Técnicas avanzadas {#técnicas-avanzadas}

### 🎯 Indexación avanzada
```python
# Indexación booleana
arr = np.array([1, -2, 3, -4, 5])
positivos = arr[arr > 0]  # [1, 3, 5]

# Indexación fancy
indices = np.array([0, 2, 4])
seleccionados = arr[indices]  # [1, 3, 5]

# Asignación condicional
arr[arr < 0] = 0  # Convierte negativos en 0
```

### 🔄 Reestructuración eficiente
```python
# Reshape sin copiar datos
matriz = np.arange(12)
vista_3x4 = matriz.reshape(3, 4)
vista_2x6 = matriz.reshape(2, 6)

# Transpose eficiente
matriz_T = matriz.T  # Vista, no copia

# Stack y concatenate
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vertical = np.vstack([a, b])    # [[1,2,3], [4,5,6]]
horizontal = np.hstack([a, b])  # [1,2,3,4,5,6]
```

### 💪 Funciones universales (ufuncs)
```python
# Operaciones elemento a elemento optimizadas
arr = np.linspace(0, np.pi, 1000)

# Todas estas son ufuncs (súper rápidas)
seno = np.sin(arr)
coseno = np.cos(arr)
exponencial = np.exp(arr)
logaritmo = np.log(arr + 1)

# Combinar ufuncs
resultado = np.sqrt(np.sin(arr)**2 + np.cos(arr)**2)  # Siempre 1
```

### 🧮 Álgebra lineal vectorizada
```python
# Multiplicación de matrices
A = np.random.rand(100, 50)
B = np.random.rand(50, 30)

# ❌ NO hagas esto
resultado_lento = np.zeros((100, 30))
for i in range(100):
    for j in range(30):
        for k in range(50):
            resultado_lento[i,j] += A[i,k] * B[k,j]

# ✅ Haz esto
resultado_rapido = A @ B  # o np.dot(A, B)
```

---

## 7. Casos prácticos del mundo real {#casos-prácticos}

### 📊 Caso 1: Análisis de series temporales
```python
# Datos de ventas diarias durante 1 año
np.random.seed(42)
ventas = np.random.normal(1000, 200, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 300

# Media móvil de 7 días (vectorizado)
def media_movil_vectorizada(data, ventana=7):
    # Crear matriz de ventanas deslizantes
    n = len(data)
    indices = np.arange(ventana)[None, :] + np.arange(n - ventana + 1)[:, None]
    return data[indices].mean(axis=1)

media_movil = media_movil_vectorizada(ventas)

# Detectar anomalías (valores > 2 desviaciones estándar)
media = ventas.mean()
std = ventas.std()
anomalias = np.abs(ventas - media) > 2 * std
dias_anomalos = np.where(anomalias)[0]
```

### 🖼️ Caso 2: Procesamiento de imágenes
```python
# Simular imagen RGB
imagen = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

# Convertir a escala de grises (vectorizado)
# Fórmula: 0.299*R + 0.587*G + 0.114*B
pesos = np.array([0.299, 0.587, 0.114])
gris = (imagen @ pesos).astype(np.uint8)

# Aplicar filtro de brillo
brillo = 50
imagen_brillante = np.clip(imagen + brillo, 0, 255).astype(np.uint8)

# Detectar bordes (simplificado)
kernel_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# Aquí usarías scipy.signal.convolve2d, pero el concepto es vectorización
```

### 📈 Caso 3: Machine Learning
```python
# Dataset de clasificación
n_muestras = 10000
n_features = 20
X = np.random.randn(n_muestras, n_features)
y = (X[:, 0] + X[:, 1] * 2 - X[:, 2] * 0.5 + np.random.randn(n_muestras) * 0.1) > 0

# Normalización vectorizada
X_normalizado = (X - X.mean(axis=0)) / X.std(axis=0)

# Cálculo de distancias (para k-NN)
def distancias_euclideas_vectorizadas(X1, X2):
    # X1: (n1, d), X2: (n2, d)
    # Resultado: (n1, n2) matriz de distancias
    X1_cuadrado = np.sum(X1**2, axis=1)[:, np.newaxis]
    X2_cuadrado = np.sum(X2**2, axis=1)[np.newaxis, :]
    X1_X2 = X1 @ X2.T
    distancias = np.sqrt(X1_cuadrado + X2_cuadrado - 2 * X1_X2)
    return distancias

# Calcular todas las distancias de una vez
distancias = distancias_euclideas_vectorizadas(X_normalizado[:100], X_normalizado[:100])
```

---

## 8. Errores comunes y cómo evitarlos {#errores-comunes}

### ❌ Error 1: Modificar vistas sin darse cuenta
```python
# PROBLEMA
arr = np.arange(10)
vista = arr[5:]
vista[0] = 999  # ¡Esto modifica arr también!

# SOLUCIÓN
copia = arr[5:].copy()
copia[0] = 999  # arr no se modifica
```

### ❌ Error 2: Broadcasting no intencional
```python
# PROBLEMA
a = np.array([[1, 2, 3]])      # shape: (1, 3)
b = np.array([[1], [2], [3]])  # shape: (3, 1)
resultado = a + b  # shape: (3, 3) - ¿Era esto lo que querías?

# SOLUCIÓN: Verificar shapes
print(f"a.shape: {a.shape}, b.shape: {b.shape}")
# Si quieres suma elemento a elemento, asegúrate de que las shapes coincidan
```

### ❌ Error 3: Uso ineficiente de memoria
```python
# PROBLEMA
matriz_grande = np.random.rand(10000, 10000)  # ~763 MB
matriz_copia = matriz_grande * 2  # Otros ~763 MB

# SOLUCIÓN: Operaciones in-place cuando sea posible
matriz_grande *= 2  # Modifica en el lugar, sin copia extra

# O usar vistas
matriz_vista = matriz_grande[::2, ::2]  # Vista, no copia
```

### ❌ Error 4: No aprovechar funciones built-in
```python
# PROBLEMA
def mi_suma(arr):
    total = 0
    for x in arr:
        total += x
    return total

# SOLUCIÓN
total = arr.sum()  # Mucho más rápido

# Otras funciones útiles:
# arr.mean(), arr.std(), arr.min(), arr.max()
# arr.argmin(), arr.argmax(), arr.argsort()
```

---

## 9. Proyecto final integrador {#proyecto-final}

### 🎯 Proyecto: Sistema de recomendación vectorizado

Vamos a construir un sistema de recomendación simple pero eficiente usando solo operaciones vectorizadas.

```python
import numpy as np

class SistemaRecomendacion:
    """Sistema de recomendación basado en similitud coseno"""
    
    def __init__(self, n_usuarios=1000, n_items=500, n_features=20):
        # Simular datos: matriz de características de items
        self.items_features = np.random.randn(n_items, n_features)
        
        # Simular preferencias de usuarios (sparse)
        self.ratings = np.zeros((n_usuarios, n_items))
        # Cada usuario califica ~50 items aleatorios
        for u in range(n_usuarios):
            items_calificados = np.random.choice(n_items, 50, replace=False)
            self.ratings[u, items_calificados] = np.random.randint(1, 6, 50)
    
    def normalizar_features(self):
        """Normaliza las características de los items"""
        # Vectorizado: normalizar cada columna
        self.items_features = (self.items_features - self.items_features.mean(axis=0)) / self.items_features.std(axis=0)
    
    def calcular_similitud_items(self):
        """Calcula matriz de similitud entre todos los items"""
        # Normalizar vectores para similitud coseno
        normas = np.linalg.norm(self.items_features, axis=1, keepdims=True)
        items_norm = self.items_features / normas
        
        # Producto matricial = similitud coseno
        self.similitud = items_norm @ items_norm.T
        
        # Poner diagonal en 0 (un item no es similar a sí mismo para recomendaciones)
        np.fill_diagonal(self.similitud, 0)
    
    def predecir_ratings(self, usuario_id, n_vecinos=20):
        """Predice ratings para items no calificados"""
        # Items que el usuario ya calificó
        items_calificados = self.ratings[usuario_id] > 0
        
        # Para cada item no calificado, usar vecinos más similares
        predicciones = np.zeros(self.ratings.shape[1])
        
        for item in range(self.ratings.shape[1]):
            if items_calificados[item]:
                predicciones[item] = self.ratings[usuario_id, item]
            else:
                # Encontrar k vecinos más similares que el usuario sí calificó
                similitudes = self.similitud[item].copy()
                similitudes[~items_calificados] = -1  # Ignorar items no calificados
                
                # Índices de los k más similares
                vecinos_idx = np.argpartition(similitudes, -n_vecinos)[-n_vecinos:]
                vecinos_idx = vecinos_idx[similitudes[vecinos_idx] > 0]
                
                if len(vecinos_idx) > 0:
                    # Predicción ponderada por similitud
                    pesos = similitudes[vecinos_idx]
                    ratings_vecinos = self.ratings[usuario_id, vecinos_idx]
                    predicciones[item] = np.average(ratings_vecinos, weights=pesos)
        
        return predicciones
    
    def recomendar_items(self, usuario_id, n_recomendaciones=10):
        """Recomienda los mejores items para un usuario"""
        predicciones = self.predecir_ratings(usuario_id)
        
        # Excluir items ya calificados
        items_no_calificados = self.ratings[usuario_id] == 0
        predicciones[~items_no_calificados] = -1
        
        # Top N items
        top_items = np.argpartition(predicciones, -n_recomendaciones)[-n_recomendaciones:]
        top_items = top_items[np.argsort(predicciones[top_items])[::-1]]
        
        return top_items, predicciones[top_items]
    
    def evaluar_sistema(self, test_size=0.2):
        """Evalúa el sistema con validación cruzada"""
        # Separar datos de prueba
        mask_test = np.random.rand(*self.ratings.shape) < test_size
        mask_test &= self.ratings > 0  # Solo items calificados
        
        ratings_train = self.ratings.copy()
        ratings_train[mask_test] = 0
        
        # Guardar ratings originales y usar ratings_train
        ratings_original = self.ratings.copy()
        self.ratings = ratings_train
        
        # Calcular similitudes con datos de entrenamiento
        self.calcular_similitud_items()
        
        # Predecir y calcular error
        errores = []
        for u in range(self.ratings.shape[0]):
            if np.any(mask_test[u]):
                predicciones = self.predecir_ratings(u)
                items_test = np.where(mask_test[u])[0]
                error = np.abs(predicciones[items_test] - ratings_original[u, items_test])
                errores.extend(error)
        
        # Restaurar ratings originales
        self.ratings = ratings_original
        
        return np.mean(errores), np.std(errores)

# Usar el sistema
sistema = SistemaRecomendacion()
sistema.normalizar_features()
sistema.calcular_similitud_items()

# Hacer recomendaciones para el usuario 0
items_recomendados, scores = sistema.recomendar_items(0)
print(f"Items recomendados: {items_recomendados}")
print(f"Scores predichos: {scores}")

# Evaluar el sistema
mae, std = sistema.evaluar_sistema()
print(f"Error absoluto medio: {mae:.3f} ± {std:.3f}")
```

### 🎉 ¡Felicitaciones!

Has completado el curso de vectorización con NumPy. Ahora puedes:

1. ✅ Identificar oportunidades de vectorización en tu código
2. ✅ Transformar bucles lentos en operaciones vectorizadas rápidas
3. ✅ Usar broadcasting para operaciones complejas
4. ✅ Aplicar técnicas avanzadas de NumPy
5. ✅ Evitar errores comunes
6. ✅ Implementar algoritmos complejos de forma eficiente

### 📚 Recursos para seguir aprendiendo

1. **Documentación oficial de NumPy**: https://numpy.org/doc/stable/
2. **NumPy para MATLAB users**: Si vienes de MATLAB
3. **From Python to NumPy** de Nicolas P. Rougier
4. **100 NumPy exercises**: Para practicar

### 💡 Consejo final

> "La vectorización no es solo sobre velocidad, es sobre pensar en términos de operaciones sobre conjuntos de datos completos en lugar de elementos individuales. Este cambio de mentalidad te hará un mejor programador científico."

---

## 🏆 Certificado de completación

**¡Felicidades!** Has dominado los conceptos de vectorización con NumPy. Ahora tienes las herramientas para escribir código Python que es:

- ⚡ 10-100x más rápido
- 📖 Más legible y mantenible
- 🎯 Más preciso y menos propenso a errores
- 💪 Capaz de manejar grandes volúmenes de datos

¡Ahora ve y vectoriza el mundo! 🚀