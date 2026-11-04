# Transcriptor de seminarios

Interfaz web (Gradio) para subir un audio desde el teléfono, transcribirlo con OpenAI Whisper en GPU y descargar la transcripción en PDF.

## Uso en Google Colab (una sola celda)

1. Activa la GPU: `Entorno de ejecución` > `Cambiar tipo de entorno de ejecución` > **T4 GPU**.
2. (Recomendado) Crea un Secret `APP_PASSWORD` en Colab (icono de llave) con acceso al cuaderno. El usuario es `seminarios`.
3. Abre [`colab_launcher.ipynb`](colab_launcher.ipynb) en Colab (o pega su celda en tu cuaderno), ajusta `REPO_URL` y ejecútala.
4. Abre el enlace `https://xxxx.gradio.live` en el teléfono.

Deja la celda en ejecución mientras uses la app. El enlace cambia en cada ejecución.

## Uso local (requiere GPU NVIDIA y ffmpeg)

```bash
pip install -r requirements.txt
APP_PASSWORD=micontraseña python app.py
```

## Variables de entorno

| Variable | Descripción | Por defecto |
|---|---|---|
| `APP_USER` | Usuario de acceso | `seminarios` |
| `APP_PASSWORD` | Contraseña; vacía = sin protección | vacía |
