"""Transcriptor de seminarios: sube un audio, transcribe con Whisper (GPU) y descarga un PDF.
Soporta traducción automática al español para seminarios en inglés mediante Gemini AI o traductor integrado.

Uso (en Colab o local con GPU):
    python app.py

Variables de entorno opcionales:
    APP_USER        usuario para proteger la interfaz (por defecto "seminarios")
    APP_PASSWORD    contraseña; si está vacía, el acceso es libre
    GEMINI_API_KEY  clave de Gemini para traducciones de máxima calidad científica al español
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


def build_pdf(entries, title, subtitle_text, pdf_path):
    """Genera el PDF con las marcas de tiempo y el texto traducido o transcrito."""
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
        Paragraph(html.escape(subtitle_text), subtitle_style),
        Spacer(1, 10),
    ]
    for tag, text in entries:
        text = text.strip()
        if not text:
            continue
        tag_html = f"<font color='#2B6CB0'><b>[{html.escape(tag)}]</b></font> " if tag else ""
        story.append(Paragraph(f"{tag_html}{html.escape(text)}", line_style))

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54,
        title=title,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def translate_with_gemini(raw_entries, api_key, progress=None):
    """Traduce segmentos usando Gemini 2.5 Flash preservando marcas de tiempo."""
    from google import genai

    client = genai.Client(api_key=api_key)
    BATCH_SIZE = 100
    translated_entries = []
    total_batches = (len(raw_entries) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(total_batches):
        batch = raw_entries[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        if progress:
            pct = 0.65 + 0.25 * ((b_idx + 1) / total_batches)
            progress(pct, desc=f"Traduciendo con Gemini AI al español ({b_idx + 1}/{total_batches})...")

        batch_lines = [f"[{tag}] {txt}" for tag, txt in batch]
        prompt = (
            "Eres un traductor experto en seminarios científicos, técnicos y académicos.\n"
            "Traduce al español la siguiente transcripción de un seminario impartido en inglés.\n\n"
            "Reglas estrictas:\n"
            "1. Mantén intacta la marca de tiempo exacta al inicio de cada línea (ejemplo: [00:01:23]).\n"
            "2. Traduce el texto manteniendo rigor en la terminología científica/técnica y naturalidad en español.\n"
            "3. Devuelve únicamente las líneas traducidas con sus marcas de tiempo, sin preámbulos, saludos ni notas.\n\n"
            + "\n".join(batch_lines)
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        lines = [ln.strip() for ln in (response.text or "").split("\n") if ln.strip()]
        for ln in lines:
            m = re.match(r"^\[([0-9:]+)\]\s*(.*)$", ln)
            if m:
                translated_entries.append((m.group(1), m.group(2).strip()))
            else:
                translated_entries.append(("", ln))

    # Si por algún motivo Gemini devolvió vacío, levantamos excepción para activar el fallback
    if not translated_entries:
        raise ValueError("Gemini no devolvió texto traducido.")

    return translated_entries, "Gemini AI (2.5 Flash)"


def translate_with_fallback(raw_entries, progress=None):
    """Traduce segmentos usando deep-translator (Google Translate) sin requerir API key."""
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target="es")
    BATCH_SIZE = 25
    translated_entries = []
    total_batches = (len(raw_entries) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(total_batches):
        batch = raw_entries[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        if progress:
            pct = 0.65 + 0.25 * ((b_idx + 1) / total_batches)
            progress(pct, desc=f"Traduciendo al español ({b_idx + 1}/{total_batches})...")

        texts = [txt for _, txt in batch]
        try:
            tr_texts = translator.translate_batch(texts)
        except Exception:
            tr_texts = []
            for t in texts:
                try:
                    tr_texts.append(translator.translate(t))
                except Exception:
                    tr_texts.append(t)

        for (tag, _), tr_t in zip(batch, tr_texts):
            translated_entries.append((tag, tr_t or ""))

    return translated_entries, "Traductor Automático Integrado"


def transcribe(file_path, title, audio_lang, target_mode, model_name, gemini_key_ui, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Primero sube un archivo de audio.")

    title = (title or "").strip() or "Transcripción de seminario"
    source_name = os.path.basename(file_path)
    stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(source_name)[0]).strip("_") or "transcripcion"

    progress(0.05, desc=f"Cargando modelo Whisper ({model_name})...")
    model = get_model(model_name)

    # 1. Transcripción con Whisper en GPU
    whisper_task = "translate" if target_mode == "to_en" else "transcribe"
    progress(0.15, desc="Transcribiendo audio con GPU... (puede tardar varios minutos)")
    start_time = time.time()

    result = model.transcribe(
        file_path,
        language=None if audio_lang == "auto" else audio_lang,
        task=whisper_task,
        fp16=(DEVICE == "cuda"),
        verbose=False,
    )
    whisper_elapsed = time.time() - start_time
    detected_lang = result.get("language", audio_lang)

    # Convertimos los segmentos originales a pares (timestamp_str, texto)
    raw_entries = [
        (fmt_time(seg["start"]), seg["text"].strip())
        for seg in result["segments"]
        if seg["text"].strip()
    ]

    out_dir = tempfile.mkdtemp()

    # Guardamos siempre la transcripción original como respaldo
    orig_txt_path = os.path.join(out_dir, f"{stem}_original_{detected_lang}.txt")
    with open(orig_txt_path, "w", encoding="utf-8") as f:
        for tag, txt in raw_entries:
            f.write(f"[{tag}] {txt}\n")

    final_entries = raw_entries
    engine_used = None
    translated_to_spanish = False

    # 2. Traducción al español si se solicitó y el audio no está ya en español
    if target_mode == "to_es":
        if detected_lang == "es":
            engine_used = "El audio ya está en español (no requiere traducción)."
        else:
            api_key = (gemini_key_ui or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
            if api_key:
                try:
                    final_entries, engine_used = translate_with_gemini(raw_entries, api_key, progress)
                    translated_to_spanish = True
                except Exception as e:
                    print(f"Aviso con Gemini ({e}); utilizando traductor alternativo...")
                    final_entries, engine_used = translate_with_fallback(raw_entries, progress)
                    engine_used += " (Fallback por error de clave Gemini)"
                    translated_to_spanish = True
            else:
                final_entries, engine_used = translate_with_fallback(raw_entries, progress)
                translated_to_spanish = True

    # 3. Generación de PDF y TXT final
    progress(0.92, desc="Generando PDF y archivos para descarga...")
    total_elapsed = time.time() - start_time

    suffix = "_espanol" if translated_to_spanish else ""
    final_pdf_path = os.path.join(out_dir, f"{stem}{suffix}.pdf")
    final_txt_path = os.path.join(out_dir, f"{stem}{suffix}.txt")

    with open(final_txt_path, "w", encoding="utf-8") as f:
        for tag, txt in final_entries:
            f.write(f"[{tag}] {txt}\n")

    if translated_to_spanish:
        doc_subtitle = (
            f"Archivo: {source_name} · Idioma original: {detected_lang.upper()} · "
            f"Traducido al español con {engine_used} · {datetime.now():%d/%m/%Y %H:%M}"
        )
    elif target_mode == "to_en":
        doc_subtitle = f"Archivo: {source_name} · Traducido a inglés con Whisper · {datetime.now():%d/%m/%Y %H:%M}"
    else:
        doc_subtitle = f"Archivo: {source_name} · Transcrito con Whisper ({detected_lang.upper()}) · {datetime.now():%d/%m/%Y %H:%M}"

    build_pdf(final_entries, title, doc_subtitle, final_pdf_path)

    # Vista previa
    preview_lines = [f"[{tag}] {txt}" for tag, txt in final_entries[:40]]
    preview = "\n".join(preview_lines)
    if len(final_entries) > 40:
        preview += f"\n\n[... mostrando 40 de {len(final_entries)} segmentos; el documento completo está en el PDF y TXT ...]"

    status = (
        f"✅ **Proceso completado en {total_elapsed / 60:.1f} min** (Whisper: {whisper_elapsed / 60:.1f} min).\n"
        f"- Idioma detectado en el audio: **{detected_lang.upper()}**\n"
    )
    if translated_to_spanish:
        status += f"- Traducción al español realizada con: **{engine_used}**\n"
    elif target_mode == "to_en":
        status += "- Traducción directa a inglés realizada con **Whisper**\n"
    else:
        status += "- Modo: **Transcripción en idioma original**\n"

    status += "📄 Descarga tu PDF y archivos de texto abajo."

    return final_pdf_path, final_txt_path, orig_txt_path, preview, status


def build_ui():
    with gr.Blocks(title="Transcriptor de seminarios") as demo:
        gr.Markdown(
            "# 🎙️ Transcriptor de seminarios\n"
            "Sube un audio (en inglés, español u otro idioma), transcríbelo con GPU y descarga el **PDF en español**."
        )
        with gr.Row():
            with gr.Column():
                audio_in = gr.File(label="Archivo de audio", file_types=AUDIO_TYPES, type="filepath")
                title_in = gr.Textbox(label="Título del documento", value="Transcripción de seminario")

                target_mode_in = gr.Radio(
                    [
                        ("Traducir al español 🇲🇽/🇪🇸 (Ideal para audios en inglés)", "to_es"),
                        ("Mantener idioma original (sin traducción)", "original"),
                        ("Traducir a inglés 🇺🇸 (Whisper)", "to_en"),
                    ],
                    value="to_es",
                    label="Resultado deseado",
                    info="Si el seminario fue en inglés, selecciona 'Traducir al español' para obtener el PDF traducido.",
                )

                audio_lang_in = gr.Dropdown(
                    [
                        ("Detectar automáticamente", "auto"),
                        ("English (Inglés)", "en"),
                        ("Español", "es"),
                        ("Français", "fr"),
                        ("Deutsch", "de"),
                        ("Português", "pt"),
                        ("Italiano", "it"),
                    ],
                    value="auto",
                    label="Idioma hablado en el audio",
                    info="Déjalo en automático o selecciona inglés si sabes que fue en ese idioma.",
                )

                with gr.Accordion("Opciones avanzadas y Clave de Gemini", open=False):
                    model_in = gr.Dropdown(
                        ["small", "medium", "large-v3"],
                        value="medium",
                        label="Modelo Whisper",
                        info="medium: buen equilibrio y rápido. large-v3: máxima precisión técnica.",
                    )
                    gemini_key_in = gr.Textbox(
                        label="Gemini API Key (opcional)",
                        type="password",
                        placeholder="Pega aquí tu clave si no la guardaste en Colab Secrets",
                        info="Si ya creaste el secret GEMINI_API_KEY en Colab, no necesitas poner nada aquí.",
                    )

                run_btn = gr.Button("Transcribir y Generar PDF", variant="primary", size="lg")

            with gr.Column():
                status_out = gr.Markdown()
                pdf_out = gr.File(label="📄 PDF (Transcripción / Traducción)")
                txt_out = gr.File(label="📝 Archivo de texto (.txt)")
                orig_txt_out = gr.File(label="🗂️ Transcripción original en inglés (.txt de respaldo)")
                preview_out = gr.Textbox(label="Vista previa del texto", lines=12, interactive=False)

        run_btn.click(
            transcribe,
            inputs=[audio_in, title_in, audio_lang_in, target_mode_in, model_in, gemini_key_in],
            outputs=[pdf_out, txt_out, orig_txt_out, preview_out, status_out],
        )
    return demo


if __name__ == "__main__":
    if DEVICE == "cpu":
        print("⚠️ No hay GPU activa: la transcripción será lenta. Activa la GPU T4 en Colab.")
    else:
        print(f"Dispositivo: CUDA ({torch.cuda.get_device_name(0)})")

    user = os.environ.get("APP_USER", "seminarios")
    password = os.environ.get("APP_PASSWORD", "")
    if not password:
        print("ℹ️ Sin contraseña: la app estará abierta a cualquiera con el enlace.")

    # Un solo trabajo a la vez para no saturar la GPU.
    build_ui().queue(default_concurrency_limit=1).launch(
        share=True,
        auth=(user, password) if password else None,
    )
