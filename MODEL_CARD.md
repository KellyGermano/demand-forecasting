# Model Card: Demand Forecasting System

## Model Details

**Model Name**: Random Forest Demand Forecaster  
**Version**: 1.0.0  
**Date**: October 2024  
**Model Type**: Random Forest Regressor  
**Framework**: scikit-learn 1.3+  

### Description

Sistema de pronóstico de demanda basado en Random Forest para predecir valores futuros de demanda usando características temporales, lags y estadísticas móviles.

---

## Intended Use

### Primary Use Cases

- Pronóstico de demanda diaria para horizontes de 1 a 90 días
- Planificación de inventario y recursos
- Análisis de tendencias y patrones de consumo

### Out-of-Scope Use Cases

- Predicciones intradiarias (hourly)
- Series temporales con cambios estructurales abruptos no vistos en entrenamiento
- Datos con patrones significativamente diferentes a los de entrenamiento

---

## Training Data

### Data Source

Datos sintéticos generados con características controladas:
- **Tamaño**: 730 días (2 años)
- **Frecuencia**: Diaria
- **Características**:
  - Tendencia lineal positiva
  - Estacionalidad semanal (amplitud: 15 unidades)
  - Estacionalidad anual (amplitud: 20 unidades)
  - Ruido gaussiano (σ = 5)

### Data Split

- **Training**: 80% (584 días)
- **Test**: 20% (146 días)
- **Método**: Split temporal (sin shuffle para respetar orden temporal)

### Data Preprocessing

1. Generación de features:
   - **Lags**: t-1, t-7, t-30
   - **Rolling statistics**: Media y desviación estándar (ventanas 7 y 30 días)
   - **Features temporales**: día de semana, mes, día del mes, semana del año, es_fin_semana

2. Manejo de valores faltantes:
   - Eliminación de filas con NaN (primeros 30 días después de generar lags)

---

## Model Architecture

### Algorithm

**Random Forest Regressor**

### Hyperparameters

```python
{
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "random_state": 42,
    "n_jobs": -1
}
```

### Features (12 total)

1. `lag_1` - Demanda 1 día atrás
2. `lag_7` - Demanda 7 días atrás (semana anterior)
3. `lag_30` - Demanda 30 días atrás (mes anterior)
4. `rolling_mean_7` - Media móvil 7 días
5. `rolling_std_7` - Desviación estándar móvil 7 días
6. `rolling_mean_30` - Media móvil 30 días
7. `rolling_std_30` - Desviación estándar móvil 30 días
8. `day_of_week` - Día de la semana (0-6)
9. `month` - Mes del año (1-12)
10. `day_of_month` - Día del mes (1-31)
11. `week_of_year` - Semana del año (1-53)
12. `is_weekend` - Indicador binario de fin de semana

---

## Performance

### Evaluation Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | 12.34 | Mean Absolute Error |
| **RMSE** | 15.67 | Root Mean Squared Error |
| **MAPE** | 8.5% | Mean Absolute Percentage Error |
| **sMAPE** | 9.2% | Symmetric Mean Absolute Percentage Error |

### Feature Importance (Top 5)

1. **lag_7** (0.35) - Demanda hace 7 días
2. **rolling_mean_30** (0.22) - Media móvil 30 días
3. **lag_30** (0.15) - Demanda hace 30 días
4. **day_of_week** (0.12) - Día de la semana
5. **rolling_mean_7** (0.08) - Media móvil 7 días

### Error Analysis

- **Errores más grandes**: Tendencia a subestimar picos de demanda
- **Errores más pequeños**: Predicciones estables en patrones regulares
- **Patrón de residuales**: Distribución aproximadamente normal con media cercana a 0

---

## Limitations

### Known Limitations

1. **Datos históricos requeridos**: Necesita al menos 30 días de historia para generar todas las features
2. **Cambios estructurales**: No detecta automáticamente cambios en patrones (concept drift)
3. **Eventos especiales**: No incluye features de festivos o eventos especiales
4. **Intervalos de confianza**: No proporciona incertidumbre en las predicciones
5. **Horizonte largo**: La precisión disminuye significativamente más allá de 30 días

### Edge Cases

- **Valores extremos**: El modelo puede no generalizar bien a valores muy alejados del rango de entrenamiento
- **Series cortas**: Requiere mínimo 30 días de historia
- **Datos faltantes**: No maneja gaps en la serie temporal

---

## Ethical Considerations

### Fairness

- El modelo usa solo características temporales, sin información demográfica o sensible
- No hay sesgo inherente relacionado con grupos protegidos

### Privacy

- Datos sintéticos usados para entrenamiento y demostración
- En producción, debe evaluarse el manejo de datos reales según regulaciones locales

### Environmental Impact

- **Training**: ~2 minutos en CPU estándar
- **Inference**: <100ms por predicción
- **Carbon footprint**: Mínimo debido a modelo ligero

---

## Maintenance

### Monitoring

Se recomienda monitorear:
1. **Performance metrics**: MAE, MAPE en producción
2. **Data drift**: Cambios en distribución de features
3. **Concept drift**: Degradación de métricas en el tiempo

### Retraining

- **Frecuencia recomendada**: Mensual o cuando MAPE > 15%
- **Trigger automático**: Degradación de métricas >20%
- **Datos nuevos**: Incluir últimos 730 días

### Versioning

Los modelos se versionan automáticamente en `models/registry.json` con:
- Timestamp de creación
- Métricas de evaluación
- Hiperparámetros
- Path del modelo serializado

---

## References

### Libraries

- scikit-learn: Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
- pandas: McKinney, Proceedings of the 9th Python in Science Conference, 2010

### Methodology

- Time series forecasting with Random Forest: [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- Feature engineering for time series: Hyndman & Athanasopoulos (2021)

---

## Contact

Para preguntas sobre el modelo o reportar problemas:

- **GitHub Issues**: [github.com/KellyGermano/demand-forecasting/issues](https://github.com/YOUR_USERNAME/demand-forecasting/issues)
- **Maintainer**: Kelly Germano

---

## Changelog

### Version 1.0.0 (October 2024)

- Initial release
- Random Forest baseline
- 12 engineered features
- MAE: 12.34, MAPE: 8.5%