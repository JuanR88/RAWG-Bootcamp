# Proyecto RAWG – API de Notebooks y Modelo de Éxito de Juegos

Este proyecto forma parte de un análisis de datos de la plataforma **RAWG** (videojuegos). A partir de un conjunto de notebooks Jupyter se realiza:

- Exploración y limpieza de datos de videojuegos.
- Entrenamiento de un modelo de **predicción de “éxito”** de juegos.
- Generación de visualizaciones y respuestas a preguntas en lenguaje natural sobre el dataset.

Toda esa lógica se expone a través de una **API REST construida con FastAPI**, de forma que cualquier persona puede:

- Lanzar el entrenamiento del modelo desde un endpoint.
- Consultar la probabilidad de éxito de un juego en función de sus géneros.
- Ejecutar notebooks de visualización y preguntas sin abrir Jupyter, solo con peticiones HTTP (por ejemplo, desde Postman).

El objetivo es convertir el trabajo de análisis en una **API consumible** y fácil de probar, tanto en local como desplegada en la nube.

Actualmente el servicio está desplegado en una instancia **EC2 de AWS** y se accede mediante la IP pública:

- `http://13.62.171.87:8000/docs`

La forma recomendada de probarlo y documentarlo es mediante **Postman**.

---


## Endpoints principales

- **GET** `/health`
  - Comprueba el estado del servidor.

- **POST** `/predict/success`
  - Predice el éxito de un juego a partir de una lista de géneros.
  - Body ejemplo:
    ```json
    {
      "genres": ["Action", "RPG"]
    }
    ```

- **POST** `/run/entrenamiento`
  - Ejecuta el notebook `Entrenamiento.ipynb` para entrenar/actualizar el modelo.

- **POST** `/run/visualizacion`
  - Ejecuta `Visualizacion.ipynb` para generar visualizaciones.
  - Body ejemplo:
    ```json
    {
      "question": "Evolución de lanzamientos",
      "min_year": 2000,
      "max_year": 2020
    }
    ```

- **POST** `/run/preguntas`
  - Ejecuta `Preguntas.ipynb` para responder preguntas sobre el dataset.
  - Body ejemplo:
    ```json
    {
      "question": "Top 10 juegos por metacritic"
    }
    ```

---

## Uso con Postman

1. Crear una colección en Postman (por ejemplo, `RAWG API`).
2. Definir una variable de entorno `base_url` con:
   - `http://13.62.171.87:8000`
3. Añadir peticiones:
   - `GET {{base_url}}/health`
   (Verifica que el servidor está funcionando)
   - `POST {{base_url}}/predict/success` con body JSON:
     ```json
     {
       "genres": ["Action", "RPG"]
     }
     ```
   - `POST {{base_url}}/run/entrenamiento`
    (Solo ejecutar para entrenar el modelo)
   - `POST {{base_url}}/run/visualizacion`
    ```json
    {
      "question": "Evolución de lanzamientos",
      "min_year": 2000,
      "max_year": 2020
    }
    ```

   - `POST {{base_url}}/run/preguntas`
    ```json
    {
      "question": "Top 10 juegos por metacritic"
    }
    ```
Con esto ya puedes probar toda la funcionalidad desde Postman.

---

## Ejecución en local (opcional)

Requisitos mínimos:

- **Python** 3.10+ (recomendado).
- Dependencias instaladas desde `requirements.txt`:

```bash
pip install -r requirements.txt
```

Para desarrollo local (no necesario si solo usas el despliegue en AWS):

```bash
python main.py
```

Por defecto se levantará en `127.0.0.1:8000` (puedes ajustar host/puerto con `uvicorn`).

---

## Estructura del repositorio (GitHub)

```text
RAWG/
├── artifacts/
│   ├── success_features.joblib
│   ├── success_model.joblib
│   └── success_stats.joblib
├── NoteBooks/   
│   ├── Entrenamiento.ipynb
│   ├── Extraccion.ipynb
│   ├── Preguntas.ipynb
│   ├── Visualizacion.ipynb
│   └── _executed/
├── Lambdas/ 
│   ├── Diaria.ipynb
│   └── CSV-RDS.ipynb
├── artifacts/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

- **`artifacts/`**
  - Carpeta donde se guardan los artefactos del modelo entrenado.
  - Contiene principalmente:
    - `success_model.joblib`: el modelo de ML entrenado que se usa en `/predict/success`.
    - `success_features.joblib`: lista de columnas/variables que el modelo espera como entrada.
    - `success_stats.joblib`: estadísticas y parámetros auxiliares (umbrales, valores por defecto, etc.).
  - Estos archivos se generan o actualizan al ejecutar el notebook de **Entrenamiento**.

- **`NoteBooks/`**
  - Carpeta con los notebooks Jupyter que concentran la lógica de negocio y el análisis de datos.
  - Algunos notebooks clave:
    - `Entrenamiento.ipynb`: entrena el modelo, calcula estadísticas y guarda los artefactos en `artifacts/`.
    - `Visualizacion.ipynb`: genera gráficas y respuestas visuales según las preguntas y rangos de años.
    - `Preguntas.ipynb`: responde preguntas sobre el dataset utilizando los datos procesados.
  - La API llama a estos notebooks para ejecutar lógica compleja sin duplicarla en código Python.

- **`Lambdas/`**
  - Carpeta con los archivos de las funciones Lambda de AWS.
  - Contiene principalmente:
    - `Diaria.ipynb`: función Lambda que extrae juegos diarios de la API RAWG y los guarda en S3.
    - `CSV-RDS.ipynb`: función Lambda que extrae datos S3 y los guarda en RDS.

- **`main.py`**
  - Archivo principal de la API **FastAPI**.
  - Define los endpoints (`/health`, `/predict/success`, `/run/entrenamiento`, `/run/visualizacion`, `/run/preguntas`).
  - Se encarga de:
    - Cargar los artefactos del modelo desde `artifacts/`.
    - Ejecutar los notebooks de `NoteBooks/` cuando se llaman los endpoints correspondientes.
    - Formatear las respuestas para que sean fáciles de consumir desde Postman u otras herramientas.

- **`requirements.txt`**
  - Lista de dependencias necesarias para ejecutar tanto la API como los notebooks.
  - Permite recrear el entorno rápidamente con: `pip install -r requirements.txt`.

En resumen, **`NoteBooks/`** contiene la lógica de análisis, **`artifacts/`** guarda el modelo ya entrenado y sus datos auxiliares, y **`main.py`** expone todo eso como una API lista para probarse con Postman.

---

## Notas

- Mantén las credenciales y claves API fuera del repositorio (usa `.env` y `.gitignore`).
- Utiliza Postman para probar rápidamente todos los endpoints sobre la IP pública `13.62.171.87:8000`.
