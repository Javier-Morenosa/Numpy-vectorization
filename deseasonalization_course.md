# Curso Completo: Desestacionalización con Descomposición Multiplicativa

## Módulo 1: Fundamentos Teóricos

### 1.1 ¿Qué es la Desestacionalización?

La desestacionalización es el proceso de eliminar patrones estacionales recurrentes de una serie temporal para revelar tendencias subyacentes y ciclos económicos. Es como "limpiar" los datos de variaciones predecibles que ocurren en períodos regulares.

**Analogía conceptual**: Imagina que estás analizando el tráfico de una autopista, pero quieres entender las tendencias de largo plazo sin que te confundan los patrones diarios (rush hours) o semanales (menos tráfico los domingos).

### 1.2 Componentes de una Serie Temporal

Toda serie temporal puede descomponerse en:

- **Tendencia (T)**: Movimiento direccional de largo plazo
- **Estacionalidad (S)**: Patrones que se repiten en períodos fijos
- **Ciclo (C)**: Fluctuaciones de mediano plazo no fijas
- **Irregular/Ruido (I)**: Variaciones aleatorias

### 1.3 Modelos de Descomposición

**Modelo Aditivo**: Y(t) = T(t) + S(t) + C(t) + I(t)
- La estacionalidad tiene amplitud constante
- Apropiado cuando las fluctuaciones estacionales son independientes del nivel de la serie

**Modelo Multiplicativo**: Y(t) = T(t) × S(t) × C(t) × I(t)
- La estacionalidad cambia proporcionalmente con el nivel de la serie
- Más común en datos económicos reales

## Módulo 2: Matemáticas de la Descomposición Multiplicativa

### 2.1 Formulación Matemática

En el modelo multiplicativo simplificado:
```
Y(t) = T(t) × S(t) × I(t)
```

Para desestacionalizar:
```
Y_desest(t) = Y(t) / S(t) = T(t) × I(t)
```

### 2.2 Estimación de Componentes

**Paso 1: Estimación de Tendencia**
Usando medias móviles centradas de orden igual al período estacional:

Para datos mensuales (período = 12):
```
T(t) = (1/12) × Σ[i=-5 to 6] Y(t+i)
```

**Paso 2: Estimación de Factor Estacional**
```
S_raw(t) = Y(t) / T(t)
```

**Paso 3: Normalización de Factores Estacionales**
Para cada mes m:
```
S(m) = Promedio{S_raw(t) : mes(t) = m}
```

Normalización para que el promedio anual sea 1:
```
S_norm(m) = S(m) / [(1/12) × Σ S(m)]
```

### 2.3 Propiedades Matemáticas

**Idempotencia**: Aplicar desestacionalización dos veces no cambia el resultado
**Reversibilidad**: Y(t) = Y_desest(t) × S(t)
**Conservación de tendencia**: La tendencia se preserva en la serie desestacionalizada

## Módulo 3: Algoritmos y Métodos

### 3.1 Método X-11/X-12-ARIMA (Censo de EE.UU.)

**Iteraciones sucesivas**:
1. Estimación inicial de tendencia (medias móviles)
2. Cálculo de factores estacionales preliminares
3. Refinamiento iterativo eliminando outliers
4. Ajuste final de factores estacionales

**Ventajas**: Robusto a outliers, ampliamente usado
**Desventajas**: Computacionalmente intensivo, "caja negra"

### 3.2 Método STL (Seasonal and Trend decomposition using Loess)

**Algoritmo**:
```
Para cada iteración k:
1. S^(k) = seasonal_smoother(Y - T^(k-1))
2. T^(k) = trend_smoother(Y - S^(k))
3. Repetir hasta convergencia
```

**Ventajas**: Flexible, maneja cambios en estacionalidad
**Desventajas**: Requiere tuning de parámetros

### 3.3 Filtros de Frecuencia

**Filtro Hodrick-Prescott**: Minimiza:
```
Σ(Y(t) - T(t))² + λΣ((T(t+1) - T(t)) - (T(t) - T(t-1)))²
```

λ controla el trade-off entre ajuste y suavidad de la tendencia.

## Módulo 4: Caso Práctico - Consumo de Gas Natural

### 4.1 Descripción del Dataset

**Características del consumo de gas**:
- Fuerte estacionalidad (mayor consumo en invierno)
- Tendencia creciente por crecimiento poblacional
- Variabilidad proporcional al nivel (modelo multiplicativo)

### 4.2 Análisis Exploratorio

**Indicadores de estacionalidad multiplicativa**:
- Coeficiente de variación estacional creciente con el tiempo
- Box plot por mes muestra dispersión creciente
- Log(serie) muestra estacionalidad más estable

### 4.3 Implementación Paso a Paso

**Datos sintéticos realistas**:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generar serie sintética de consumo de gas
np.random.seed(42)
t = np.arange(1, 121)  # 10 años, datos mensuales
tendencia = 100 + 0.5 * t + 0.01 * t**2
estacional = np.tile([1.4, 1.3, 1.1, 0.9, 0.7, 0.6, 
                     0.6, 0.7, 0.8, 1.0, 1.2, 1.3], 10)
ruido = np.random.lognormal(0, 0.05, 120)
consumo_gas = tendencia * estacional * ruido
```

**Descomposición multiplicativa manual**:
```python
def descomposicion_multiplicativa(serie, periodo=12):
    n = len(serie)
    
    # 1. Calcular tendencia con medias móviles centradas
    tendencia = np.full(n, np.nan)
    for i in range(periodo//2, n - periodo//2):
        ventana = serie[i-periodo//2:i+periodo//2+1]
        tendencia[i] = np.mean(ventana)
    
    # 2. Calcular ratios estacionales
    ratios = serie / tendencia
    
    # 3. Estimar factores estacionales
    factores_est = np.full(periodo, np.nan)
    for mes in range(periodo):
        indices_mes = [i for i in range(mes, n, periodo) 
                      if not np.isnan(ratios[i])]
        if indices_mes:
            factores_est[mes] = np.median([ratios[i] for i in indices_mes])
    
    # 4. Normalizar factores (promedio = 1)
    factores_est = factores_est / np.mean(factores_est)
    
    # 5. Aplicar factores a toda la serie
    factores_completos = np.tile(factores_est, n//periodo + 1)[:n]
    
    # 6. Desestacionalizar
    serie_desest = serie / factores_completos
    
    return {
        'original': serie,
        'tendencia': tendencia,
        'estacional': factores_completos,
        'desestacionalizada': serie_desest,
        'residuos': serie / (tendencia * factores_completos)
    }
```

### 4.4 Validación de Resultados

**Tests estadísticos**:
1. **Test de estacionalidad** (antes vs después):
   - Kruskal-Wallis por meses
   - Análisis de Fourier (densidad espectral)

2. **Calidad de la descomposición**:
   - R² entre original y componentes reconstruidos
   - Análisis de residuos (normalidad, autocorrelación)

**Métricas de evaluación**:
```python
def evaluar_desestacionalizacion(original, desest, factores):
    # Reconstrucción
    reconstruida = desest * factores
    
    # Métricas
    mse = np.mean((original - reconstruida)**2)
    mape = np.mean(np.abs((original - reconstruida) / original)) * 100
    r2 = 1 - np.var(original - reconstruida) / np.var(original)
    
    return {'MSE': mse, 'MAPE': mape, 'R²': r2}
```

## Módulo 5: Aspectos Avanzados

### 5.1 Tratamiento de Outliers

**Detección**:
- Outliers aditivos: Afectan observaciones individuales
- Outliers de nivel: Cambios permanentes en el nivel
- Outliers estacionales: Cambios en patrones estacionales

**Métodos de detección**:
```python
def detectar_outliers_iqr(serie_desest, factor=1.5):
    Q1 = np.percentile(serie_desest, 25)
    Q3 = np.percentile(serie_desest, 75)
    IQR = Q3 - Q1
    limite_inf = Q1 - factor * IQR
    limite_sup = Q3 + factor * IQR
    return (serie_desest < limite_inf) | (serie_desest > limite_sup)
```

### 5.2 Cambios Estructurales

**Detección de cambios**:
- Test de Chow para cambios en tendencia
- CUSUM para cambios graduales
- Análisis de ventana deslizante para cambios en estacionalidad

### 5.3 Descomposición Multi-frecuencia

Para series con múltiples patrones estacionales:
```
Y(t) = T(t) × S_anual(t) × S_semanal(t) × I(t)
```

## Módulo 6: Implementación Computacional

### 6.1 Bibliotecas Especializadas

**Python**:
- `statsmodels.tsa.seasonal.seasonal_decompose()`
- `statsmodels.tsa.x13.x13_arima_analysis()`
- `sktime` para métodos avanzados

**R**:
- `decompose()` y `stl()`
- `seasonal` package (interface a X-13ARIMA-SEATS)
- `forecast` package

### 6.2 Consideraciones de Performance

**Complejidad temporal**:
- Medias móviles: O(n×p) donde p es el período
- STL: O(n×k) donde k es número de iteraciones
- X-13: O(n×log(n)) por componente FFT

**Optimizaciones**:
- Vectorización de operaciones
- Paralelización para múltiples series
- Algoritmos online para datos streaming

### 6.3 Implementación Robusta

```python
class DesestacionalizadorMultiplicativo:
    def __init__(self, periodo=12, metodo='medias_moviles'):
        self.periodo = periodo
        self.metodo = metodo
        self.factores_estacionales = None
        self.tendencia = None
        
    def fit(self, serie):
        """Estima factores estacionales"""
        if self.metodo == 'medias_moviles':
            self._fit_medias_moviles(serie)
        elif self.metodo == 'stl':
            self._fit_stl(serie)
        return self
    
    def transform(self, serie):
        """Aplica desestacionalización"""
        if self.factores_estacionales is None:
            raise ValueError("Debe ajustar el modelo primero")
        
        factores = np.tile(self.factores_estacionales, 
                          len(serie)//self.periodo + 1)[:len(serie)]
        return serie / factores
    
    def fit_transform(self, serie):
        """Ajusta y transforma en un paso"""
        return self.fit(serie).transform(serie)
    
    def inverse_transform(self, serie_desest):
        """Revierte la desestacionalización"""
        factores = np.tile(self.factores_estacionales, 
                          len(serie_desest)//self.periodo + 1)[:len(serie_desest)]
        return serie_desest * factores
```

## Módulo 7: Casos de Uso y Aplicaciones

### 7.1 Análisis Económico

**PIB estacional vs tendencial**:
- Decisiones de política monetaria
- Comparaciones internacionales
- Detección de recesiones

### 7.2 Forecasting

**Ventajas de usar datos desestacionalizados**:
- Modelos más simples y estables
- Mejor identificación de puntos de cambio
- Horizontes de predicción más largos

**Proceso**:
1. Desestacionalizar serie histórica
2. Modelar/predecir serie desestacionalizada
3. Re-estacionalizar predicciones

### 7.3 Control de Calidad Industrial

**Monitoreo de procesos**:
- Separar variaciones estacionales de problemas reales
- Gráficos de control más sensibles
- Detección temprana de degradación

## Módulo 8: Evaluación y Diagnósticos

### 8.1 Pruebas de Calidad

**Residual seasonality test**:
```python
def test_estacionalidad_residual(serie_desest, periodo=12):
    """Test de Kruskal-Wallis para estacionalidad residual"""
    from scipy.stats import kruskal
    
    grupos = [serie_desest[i::periodo] for i in range(periodo)]
    estadistico, p_valor = kruskal(*grupos)
    
    return {
        'estadistico': estadistico,
        'p_valor': p_valor,
        'estacionalidad_residual': p_valor < 0.05
    }
```

**Estabilidad de factores estacionales**:
- Rolling window estimation
- Test de cambio estructural en factores

### 8.2 Diagnósticos Visuales

**Gráficos esenciales**:
1. Serie original vs desestacionalizada
2. Factores estacionales por período
3. Residuos vs tiempo (detectar heterocedasticidad)
4. Q-Q plot de residuos (normalidad)
5. Función de autocorrelación de residuos

### 8.3 Métricas de Performance

**Para series de validación**:
- Mean Absolute Percentage Error (MAPE)
- Mean Absolute Scaled Error (MASE)
- Symmetric MAPE (sMAPE)

## Módulo 9: Ejercicios Prácticos

### Ejercicio 1: Implementación Básica
Implementa la descomposición multiplicativa desde cero para una serie de ventas retail con estacionalidad navideña.

### Ejercicio 2: Comparación de Métodos
Compara X-11, STL y descomposición clásica en términos de:
- Calidad de ajuste
- Robustez a outliers
- Tiempo computacional

### Ejercicio 3: Series Múltiples
Desarrolla un pipeline para desestacionalizar simultáneamente múltiples series relacionadas (ej: consumo eléctrico por región).

### Ejercicio 4: Detección de Anomalías
Usa desestacionalización como preprocesamiento para un sistema de detección de anomalías en tiempo real.

## Módulo 10: Recursos y Referencias

### 10.1 Literatura Fundamental

**Papers clásicos**:
- Shiskin et al. (1967) "The X-11 Variant of the Census Method II Seasonal Adjustment Program"
- Cleveland et al. (1990) "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"
- Findley et al. (1998) "New Capabilities and Methods of the X-12-ARIMA Seasonal-Adjustment Program"

**Libros**:
- "Time Series Analysis" - Hamilton (1994)
- "Forecasting: Principles and Practice" - Hyndman & Athanasopoulos
- "The Analysis of Economic Time Series" - Granger & Newbold

### 10.2 Software y Herramientas

**Oficiales**:
- X-13ARIMA-SEATS (US Census Bureau)
- TRAMO-SEATS (Banco de España)
- JDemetra+ (Eurostat)

**Open Source**:
- Python: statsmodels, seasonal, sktime
- R: seasonal, x12, forecast
- Julia: StateSpaceModels.jl

### 10.3 Datasets de Práctica

1. **Consumo energético**: Datos horarios/mensuales de utilities
2. **Turismo**: Llegadas mensuales por país (UNWTO)
3. **Retail**: Ventas mensuales por sector (FRED)
4. **Transporte**: Pasajeros aéreos (classic airline dataset)

---

## Proyecto Final: Sistema de Desestacionalización Integral

**Objetivo**: Construir un sistema completo que:

1. **Ingesta** múltiples series temporales
2. **Detecta** automáticamente el tipo de estacionalidad
3. **Aplica** el método óptimo de desestacionalización
4. **Evalúa** la calidad de los resultados
5. **Monitor** para detectar cambios en patrones estacionales
6. **Visualiza** resultados de forma interactiva

**Criterios de evaluación**:
- Correctitud de implementación matemática
- Eficiencia computacional
- Robustez ante casos edge
- Calidad de visualizaciones
- Documentación y testing

**Entregables**:
- Código fuente documentado
- Notebook con casos de uso
- Dashboard interactivo
- Reporte técnico con benchmarks

---

*Este curso te ha llevado desde los fundamentos teóricos hasta implementaciones prácticas avanzadas. La desestacionalización es una herramienta poderosa que requiere comprensión tanto matemática como intuición práctica sobre los datos. ¡Ahora tienes las bases para aplicarla efectivamente en proyectos reales!*