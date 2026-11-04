"""Transcriptor de seminarios: sube un audio, transcribe con Whisper (GPU) y descarga un PDF.

Uso (en Colab o local con GPU):
    python app.py

Variables de entorno opcionales:
    APP_USER      usuario para proteger la interfaz (por defecto "seminarios")
    APP_PASSWORD  contraseña; si está vacía, el acceso es libre
"""
import html
import os
import re
import tempfile
import time
from datetime import datetime

import gradio as gr
import torch
import whisper
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_TYPES = ["audio", ".m4a", ".mp3", ".wav", ".aac", ".ogg", ".opus", ".flac", ".mp4", ".amr", ".3gp"]

# Solo se mantiene un modelo en memoria para no agotar la VRAM al cambiar de tamaño.
_model_cache = {}


def get_model(name):
    if name not in _model_cache:
        _model_cache.clear()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        _model_cache[name] = whisper.load_model(name, device=DEVICE)
    return _model_cache[name]


def fmt_time(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def build_pdf(segments, title, source_name, pdf_path):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=colors.HexColor("#1A365D"), spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "SubTitleStyle", parent=styles["Normal"], fontName="Helvetica-Oblique",
        fontSize=9, leading=13, textColor=colors.HexColor("#718096"), spaceAfter=15,
    )
    line_style = ParagraphStyle(
        "LineStyle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=colors.HexColor("#2D3748"), spaceAfter=6,
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#718096"))
        canvas.drawRightString(letter[0] - 54, 30, f"Página {doc.page}")
        canvas.restoreState()

    story = [
        Paragraph(html.escape(title), title_style),
        Paragraph(
            f"Archivo: {html.escape(source_name)} · Generado el "
            f"{datetime.now():%d/%m/%Y %H:%M} con OpenAI Whisper",
            subtitle_style,
        ),
        Spacer(1, 10),
    ]
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        story.append(Paragraph(
            f"<font color='#2B6CB0'><b>[{fmt_time(seg['start'])}]</b></font> {html.escape(text)}",
            line_style,
        ))

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54,
        title=title,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def transcribe(file_path, title, model_name, language, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Primero sube un archivo de audio.")

    title = (title or "").strip() or "Transcripción de seminario"
    source_name = os.path.basename(file_path)
    stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(source_name)[0]).strip("_") or "transcripcion"

    progress(0.05, desc=f"Cargando modelo Whisper ({model_name})...")
    model = get_model(model_name)

    progress(0.15, desc="Transcribiendo... (puede tardar varios minutos)")
    start = time.time()
    # Whisper lee directamente m4a/mp3/wav/etc. mediante ffmpeg (ya incluido en Colab).
    result = model.transcribe(
        file_path,
        language=None if language == "auto" else language,
        fp16=(DEVICE == "cuda"),
        verbose=False,
    )
    elapsed = time.time() - start

    progress(0.9, desc="Generando PDF...")
    out_dir = tempfile.mkdtemp()
    txt_path = os.path.join(out_dir, f"{stem}.txt")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")

    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(f"[{fmt_time(seg['start'])}] {seg['text'].strip()}\n")

    build_pdf(result["segments"], title, source_name, pdf_path)

    preview = result["text"].strip()
    if len(preview) > 3000:
        preview = preview[:3000] + "\n[... vista previa recortada; el texto completo está en el PDF ...]"
    status = (
        f"✅ Listo en {elapsed / 60:.1f} min. Idioma: **{result.get('language', language)}**. "
        "Descarga tus archivos abajo."
    )
    return pdf_path, txt_path, preview, status


def build_ui():
    with gr.Blocks(title="Transcriptor de seminarios") as demo:
        gr.Markdown("# 🎙️ Transcriptor de seminarios\nSube el audio, espera la transcripción y descarga el PDF.")
        with gr.Row():
            with gr.Column():
                audio_in = gr.File(label="Audio del seminario", file_types=AUDIO_TYPES, type="filepath")
                title_in = gr.Textbox(label="Título del documento", value="Transcripción de seminario")
                with gr.Accordion("Opciones avanzadas", open=False):
                    model_in = gr.Dropdown(
                        ["small", "medium", "large-v3"], value="medium", label="Modelo Whisper",
                        info="medium: buen equilibrio. large-v3: más preciso pero más lento.",
                    )
                    lang_in = gr.Dropdown(
                        [("Español", "es"), ("English", "en"), ("Detectar automáticamente", "auto")],
                        value="es", label="Idioma del audio",
                    )
                run_btn = gr.Button("Transcribir", variant="primary")
            with gr.Column():
                status_out = gr.Markdown()
                pdf_out = gr.File(label="📄 PDF de la transcripción")
                txt_out = gr.File(label="📝 Texto plano (.txt)")
                preview_out = gr.Textbox(label="Vista previa", lines=12, interactive=False)

        run_btn.click(
            transcribe,
            inputs=[audio_in, title_in, model_in, lang_in],
            outputs=[pdf_out, txt_out, preview_out, status_out],
        )
    return demo


if __name__ == "__main__":
    if DEVICE == "cpu":
        print("⚠️ No hay GPU activa: la transcripción será MUY lenta. Activa la T4 en Colab.")
    else:
        print(f"Dispositivo: CUDA ({torch.cuda.get_device_name(0)})")

    user = os.environ.get("APP_USER", "seminarios")
    password = os.environ.get("APP_PASSWORD", "")
    if not password:
        print("⚠️ Sin contraseña: cualquiera con el enlace podrá usar tu GPU.")

    # Un solo trabajo a la vez para no saturar la GPU.
    build_ui().queue(default_concurrency_limit=1).launch(
        share=True,
        auth=(user, password) if password else None,
    )
