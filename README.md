# Clasificador de expresiones faciales

Esta rama contiene un pipeline reproducible para clasificar ocho expresiones
faciales: `anger`, `contempt`, `disgust`, `fear`, `happy`, `neutral`, `sad` y
`surprise`.

Parte del notebook base y de los conceptos del material de Ringa Tech, pero se
adapta a un problema multiclase: salida `softmax`, pérdida de entropía cruzada
categórica, aumento de datos adecuado para rostros y evaluación independiente.

## Datos

El archivo `dataset_modelo.zip` no se versiona porque supera los 2 GB. Debe
extraerse de modo que la estructura resultante sea la siguiente:

```text
dataset_modelo/
├── Train/<clase>/...
├── Validation/<clase>/...
└── Test/<clase>/...
```

El conjunto entregado ya contiene las tres particiones y 8 clases. No se deben
mezclar ni volver a dividir: `Train` ajusta los pesos, `Validation` guía el
entrenamiento y `Test` se reserva para la medición final.

Para extraerlo de forma segura:

```powershell
python scripts/extract_dataset.py `
  --zip C:\ruta\dataset_modelo.zip `
  --destination data
```

El directorio de datos resultante será `data/dataset_modelo`.

## Entrenamiento

```powershell
python -m pip install -r requirements.txt
python src/train_expression_classifier.py `
  --data-dir data/dataset_modelo `
  --output-dir artifacts/expressions `
  --export-tflite
```

El pipeline usa MobileNetV2 preentrenada para evitar entrenar una CNN desde
cero. Primero entrena el clasificador con la base congelada y después ajusta
las últimas capas con una tasa de aprendizaje menor. Los aumentos sólo usan
transformaciones plausibles para rostros; se excluye el volteo vertical porque
crea ejemplos irreales.

El directorio de salida incluye:

- `best_model.keras`: mejor modelo según `val_accuracy`.
- `metrics.json`: clases, métricas de prueba y matriz de confusión.
- `history.csv` y `logs/`: trazas para TensorBoard.
- `expression_classifier.tflite`: modelo portable si se usa `--export-tflite`.

También hay una guía ejecutable en
`notebooks/Entrenamiento_Expresiones_Faciales.ipynb`.
