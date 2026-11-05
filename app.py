"""Transcriptor y Analizador de Seminarios:
1. Sube un audio, transcribe con Whisper (GPU) y traduce a español.
2. O sube un PDF/TXT generado previamente para reconstruir el evento con IA, generar resumen y rescatar la sesión de preguntas (Q&A).
3. Agrega contexto adicional: correos de anuncio, abstracts, notas o PDFs de artículos/capítulos de apoyo.

Uso en Google Colab:
    python app.py
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


def extract_text_from_file(file_path):
    """Extrae texto de un archivo PDF o TXT."""
    if not file_path:
        return ""
    real_path = file_path if isinstance(file_path, str) else getattr(file_path, "name", str(file_path))
    if not os.path.exists(real_path):
        return ""
    ext = os.path.splitext(real_path)[1].lower()
    if ext == ".txt" or ext == ".md":
        try:
            with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            return f"[Error leyendo TXT: {e}]"
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(real_path)
            pages = []
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    pages.append(t)
            return "\n\n".join(pages)
        except Exception as e:
            return f"[Error extrayendo texto del PDF: {e}]"
    return ""


# ---------------------------------------------------------------------------
# GENERACIÓN DE PDFS (REPORTLAB)
# ---------------------------------------------------------------------------


def build_transcript_pdf(entries, title, subtitle_text, pdf_path):
    """Genera el PDF con las marcas de tiempo y el texto transcrito/traducido."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "SubTitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#718096"),
        spaceAfter=15,
    )
    line_style = ParagraphStyle(
        "LineStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
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
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
        title=title,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def format_inline_markdown(text):
    """Convierte **negrita**, *cursiva* y `código` en etiquetas compatibles con ReportLab."""
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"__(.+?)__", r"<b>\1</b>", safe)
    safe = re.sub(r"\*(.+?)\*", r"<i>\1</i>", safe)
    safe = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", safe)
    safe = re.sub(r"`(.+?)`", r"<font face='Courier' color='#553C9A'>\1</font>", safe)
    return safe


def safe_paragraph(formatted_html, style, raw_fallback=None):
    """Crea un Paragraph de ReportLab asegurando que errores de parseo no tiren la construcción."""
    try:
        return Paragraph(formatted_html, style)
    except Exception:
        fallback = raw_fallback or re.sub(r"<[^>]+>", "", formatted_html)
        return Paragraph(html.escape(fallback), style)


def build_analysis_pdf(md_text, doc_title, pdf_path):
    """Construye un PDF formal y elegante a partir del Markdown del informe de reconstrucción."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#718096"),
        spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )
    h3_style = ParagraphStyle(
        "H3Style",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=16,
        firstLineIndent=-10,
        spaceAfter=4,
    )
    quote_style = ParagraphStyle(
        "QuoteStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        leftIndent=16,
        spaceBefore=4,
        spaceAfter=6,
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#718096"))
        canvas.drawRightString(letter[0] - 54, 30, f"Página {doc.page}")
        canvas.drawString(54, 30, "Informe Técnico y Reconstrucción de Seminario · Transcrip AI")
        canvas.restoreState()

    story = [
        safe_paragraph(html.escape(doc_title), title_style),
        safe_paragraph(
            f"Informe Ejecutivo y Reconstrucción Académica · Generado el {datetime.now():%d/%m/%Y %H:%M} con IA",
            subtitle_style,
        ),
        Spacer(1, 6),
    ]

    for line in md_text.split("\n"):
        raw = line.strip()
        if not raw:
            story.append(Spacer(1, 4))
            continue

        if raw.startswith("## "):
            h_text = raw[3:].strip().replace("**", "").replace("*", "")
            story.append(Spacer(1, 6))
            story.append(safe_paragraph(html.escape(h_text), h2_style))
        elif raw.startswith("### "):
            h_text = raw[4:].strip().replace("**", "").replace("*", "")
            story.append(safe_paragraph(html.escape(h_text), h3_style))
        elif raw.startswith("# "):
            h_text = raw[2:].strip().replace("**", "").replace("*", "")
            story.append(safe_paragraph(html.escape(h_text), title_style))
        elif raw.startswith("- ") or raw.startswith("* ") or re.match(r"^\d+\.\s+", raw):
            content = re.sub(r"^[-*]\s+|\d+\.\s+", "", raw)
            formatted = format_inline_markdown(content)
            story.append(safe_paragraph(f"• {formatted}", bullet_style, raw_fallback=f"• {content}"))
        elif raw.startswith(">"):
            content = raw.lstrip("> ").strip()
            formatted = format_inline_markdown(content)
            story.append(safe_paragraph(formatted, quote_style, raw_fallback=content))
        else:
            formatted = format_inline_markdown(raw)
            story.append(safe_paragraph(formatted, body_style, raw_fallback=raw))

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
        title=doc_title,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------------------
# PROMPT Y LLM PARA RECONSTRUCCIÓN Y ANÁLISIS
# ---------------------------------------------------------------------------

RECONSTRUCTION_PROMPT = """Eres un investigador y analista académico experto en síntesis y reconstrucción de seminarios de investigación, conferencias científicas y clases universitarias avanzadas.

Tu objetivo es procesar la transcripción de un evento académico y generar un informe técnico de altísimo nivel.
A menudo las grabaciones presentan deficiencias reales:
- La persona que graba llega tarde (se pierde la bienvenida, la presentación del ponente o la introducción).
- Se corta la grabación al inicio o al final.
- Hay baches inaudibles o acústica deficiente.
Tu tarea es subsanar esas fallas combinando deductivamente la transcripción con el contexto disponible (correos de anuncio, abstract, diapositivas, artículos de referencia).

Debes generar tu respuesta ÚNICAMENTE en formato Markdown con las siguientes 4 secciones principales (utiliza encabezados '##'):

## 1. Ficha Técnica y Diagnóstico de la Grabación
- **Título formal del seminario:** Deduce o valida el título más adecuado.
- **Ponente / Expositor:** Nombre, grado académico e institución/laboratorio de procedencia.
- **Área temática y Palabras clave:** 5 a 8 términos clave especializados.
- **Diagnóstico del registro:** Diagnóstico explícito del estado de la grabación (ej: 'Grabación completa', 'Grabación con inicio tardío - faltó la introducción original y fue reconstruida con base en el anuncio', 'Cortes intermedios').

## 2. Reconstrucción Integral del Seminario (Hilo Conductor)
- Reconstruye la totalidad de la exposición en orden cronológico, fluido y lógico.
- **Si el audio empezó a mitad de camino:** utiliza el correo de anuncio, el título y el desarrollo de la charla para reconstruir la introducción ausente, el planteamiento del problema, los antecedentes y la motivación, señalándolo explícitamente con la etiqueta *[Reconstrucción contextual a partir de anuncio/referencias: ...]* para distinguir lo deducido de lo grabado.
- Desarrolla a profundidad los temas expuestos:
  * Motivación, problemática y estado del arte.
  * Metodología experimental, diseño, reactores, técnicas analíticas (ej. CFD, HPLC, espectrofotometría, electroquímica, etc.).
  * Resultados principales, discusión de datos, gráficas explicadas y hallazgos centrales.
  * Conclusiones del expositor y aplicaciones prácticas o futuras líneas de trabajo.
- Incluye términos técnicos rigurosos, nombres de reactivos/compuestos, valores numéricos y relaciones causa-efecto explicadas por el ponente.

## 3. Resumen Ejecutivo y Puntos Rescatables ('Key Takeaways')
- **Resumen ejecutivo:** Síntesis clara y fluida de 3 a 4 párrafos de alto impacto.
- **Key Takeaways (Puntos rescatables):** Lista de 6 a 10 ideas o conclusiones fundamentales que cualquier asistente debe conservar en sus notas.
- **Hallazgos sorprendentes o advertencias metodológicas:** Aspectos críticos, limitaciones o recomendaciones destacadas por el ponente durante la sesión.

## 4. Sesión de Preguntas y Respuestas (Q&A) y Debate
- Si la transcripción incluye una sesión final de preguntas o intervenciones de los asistentes:
  * Desglosa cada pregunta formulada por el público o moderadores.
  * Sintetiza la respuesta del ponente, destacando si hubo debates, consensos o dudas abiertas.
- Si NO se grabó sesión de preguntas (o la grabación terminó antes):
  * Indica: *"No se registró sesión de preguntas en el audio disponible."*
  * Añade la sección: *"Preguntas críticas sugeridas para el trabajo presentado"*, con 3 o 4 preguntas de alto nivel que un colega o evaluador haría a esta investigación, junto con la respuesta o hipótesis probable basada en la literatura y los datos del seminario.

IMPORTANTE: Escribe en un español académico impecable, formal y sin preámbulos conversacionales; inicia directamente con el encabezado.
"""


def call_llm_for_reconstruction(
    transcription_text,
    context_text,
    context_files,
    title,
    model_choice,
    api_key_override=None,
    progress=None,
):
    """Envía la transcripción y el contexto al LLM para generar la reconstrucción y el resumen."""
    # 1. Armar contexto adicional
    context_blocks = []
    if context_text and context_text.strip():
        context_blocks.append(f"=== CORREO DE ANUNCIO / NOTAS / ABSTRACT ===\n{context_text.strip()}")

    if context_files:
        files_list = context_files if isinstance(context_files, list) else [context_files]
        for cf in files_list:
            if cf:
                cf_path = cf if isinstance(cf, str) else getattr(cf, "name", str(cf))
                bname = os.path.basename(cf_path)
                txt = extract_text_from_file(cf_path)
                if txt:
                    context_blocks.append(f"=== ARCHIVO DE APOYO ({bname}) ===\n{txt[:45000]}")

    full_context_str = "\n\n".join(context_blocks) if context_blocks else "[No se proporcionó contexto adicional]"

    user_prompt = (
        f"# TÍTULO O TEMA SUGERIDO: {title}\n\n"
        f"# CONTEXTO ADICIONAL PROVISTO:\n{full_context_str}\n\n"
        f"# TRANSCRIPCIÓN DEL EVENTO:\n{transcription_text}\n\n"
        "Genera ahora la Ficha Técnica, la Reconstrucción Integral, el Resumen Ejecutivo y el Análisis de la Sesión de Preguntas (Q&A) siguiendo tus instrucciones."
    )

    if "Local" in model_choice or "Qwen" in model_choice:
        if progress:
            progress(0.15, desc="Cargando modelo local Qwen 2.5 3B en GPU (sin APIs externas)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_id = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        messages = [
            {"role": "system", "content": RECONSTRUCTION_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        # Proteger la GPU T4 de OOM por atención cuadrática en textos gigantes
        if model_inputs.input_ids.shape[1] > 7500:
            model_inputs["input_ids"] = model_inputs["input_ids"][:, :7500]
            if "attention_mask" in model_inputs:
                model_inputs["attention_mask"] = model_inputs["attention_mask"][:, :7500]

        if progress:
            progress(0.45, desc="Generando reconstrucción e informe en tu GPU local...")

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            temperature=0.3,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response_text, "Qwen 2.5 3B (Local en GPU T4)"

    elif "Groq" in model_choice or "Llama" in model_choice:
        from groq import Groq

        api_key = (
            (api_key_override or "").strip()
            or os.environ.get("GROQ_API_KEY", "").strip()
            or os.environ.get("GROQ", "").strip()
            or os.environ.get("groq_api_key", "").strip()
        )
        if not api_key:
            raise gr.Error(
                "No se encontró la clave de Groq. Asegúrate de que el secreto en Colab se llame 'GROQ_API_KEY' "
                "(con el interruptor activado) o pégala directamente en el campo de texto."
            )

        client = Groq(api_key=api_key)
        for groq_model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            try:
                if progress:
                    progress(0.4, desc=f"Reconstruyendo con {groq_model} en Groq...")
                response = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": RECONSTRUCTION_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                return response.choices[0].message.content or "", f"Meta ({groq_model} vía Groq)"
            except Exception as e:
                print(f"Aviso Groq {groq_model}: {e}")
                continue
        raise gr.Error("No se pudo completar con Groq.")

    elif "Gemini" in model_choice:
        from google import genai

        api_key = (api_key_override or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise gr.Error(
                "Se requiere una API Key de Gemini. Agrégala en Colab Secrets (GEMINI_API_KEY) "
                "o pégala en el campo de texto."
            )

        client = genai.Client(api_key=api_key)
        last_err = None
        for attempt in range(3):
            try:
                if progress:
                    progress(0.3 + attempt * 0.15, desc=f"Conectando con Gemini 3.8 Flash (intento {attempt+1}/3)...")
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[RECONSTRUCTION_PROMPT + "\n\n" + user_prompt],
                )
                if response.text and response.text.strip():
                    return response.text.strip(), "Google Gemini 3.8 Flash"
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
                    wait_sec = 3 * (attempt + 1)
                    print(f"Gemini 3.8 saturado (503). Reintentando en {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                else:
                    raise e
        raise gr.Error("Google Gemini 3.8 sigue saturado por alta demanda (503). Por favor selecciona 'Modelo Local en GPU Colab' para procesar sin servidores externos.")

    else:
        raise gr.Error(f"Modelo no soportado: {model_choice}")


def run_reconstruction_from_file(
    file_path,
    context_text,
    context_files,
    title,
    model_choice,
    api_key_override,
    progress=gr.Progress(),
):
    """Pestaña 1: Procesa una transcripción ya existente (PDF o TXT)."""
    if not file_path:
        raise gr.Error("Primero sube un archivo PDF o TXT con la transcripción previa.")

    title = (title or "").strip() or "Seminario Académico"
    file_real_path = file_path if isinstance(file_path, str) else getattr(file_path, "name", str(file_path))
    source_name = os.path.basename(file_real_path)
    stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(source_name)[0]).strip("_") or "seminario"

    progress(0.1, desc="Extrayendo texto de la transcripción previa...")
    transcript_text = extract_text_from_file(file_real_path)
    if not transcript_text or len(transcript_text.strip()) < 30:
        raise gr.Error("El archivo subido está vacío o no se pudo extraer texto legible.")

    start_time = time.time()
    md_result, engine_name = call_llm_for_reconstruction(
        transcription_text=transcript_text,
        context_text=context_text,
        context_files=context_files,
        title=title,
        model_choice=model_choice,
        api_key_override=api_key_override,
        progress=progress,
    )
    elapsed = time.time() - start_time

    progress(0.85, desc="Generando PDF del informe ejecutivo...")
    out_dir = tempfile.mkdtemp()
    out_pdf_path = os.path.join(out_dir, f"{stem}_informe_reconstruido.pdf")
    out_md_path = os.path.join(out_dir, f"{stem}_informe_reconstruido.md")

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_result)

    build_analysis_pdf(md_result, title, out_pdf_path)

    word_count = len(transcript_text.split())
    status_msg = (
        f"✅ **Reconstrucción e Informe completados en {elapsed:.1f} segundos**.\n\n"
        f"- **Motor de IA utilizado:** {engine_name}\n"
        f"- **Archivo analizado:** `{source_name}` (~{word_count} palabras)\n"
        f"- **Documentos de apoyo procesados:** {'Sí' if context_files or (context_text and context_text.strip()) else 'No proporcionados'}\n\n"
        "📄 **Descarga tu PDF y archivo Markdown abajo**, o consulta el informe directamente en pantalla."
    )
    return out_pdf_path, out_md_path, md_result, status_msg


# ---------------------------------------------------------------------------
# TRADUCCIÓN (WHISPER RAW SEGMENTS)
# ---------------------------------------------------------------------------


def translate_with_gemini(raw_entries, api_key, progress=None):
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
            "3. Devuelve únicamente las líneas traducidas con sus marcas de tiempo, sin preámbulos ni notas.\n\n"
            + "\n".join(batch_lines)
        )

        candidates = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-3.8-flash"]
        response = None
        used_model = "gemini-1.5-flash"
        for mod in candidates:
            try:
                response = client.models.generate_content(
                    model=mod,
                    contents=prompt,
                )
                used_model = mod
                break
            except Exception as e:
                err_str = str(e).lower()
                if any(c in err_str for c in ["503", "unavailable", "high demand", "404", "not_found", "no longer available", "429"]):
                    time.sleep(1.5)
                    continue
                raise e

        if response is None:
            raise RuntimeError("No se pudo conectar a ningún modelo de Gemini disponible.")

        lines = [ln.strip() for ln in (response.text or "").split("\n") if ln.strip()]
        for ln in lines:
            m = re.match(r"^\[([0-9:]+)\]\s*(.*)$", ln)
            if m:
                translated_entries.append((m.group(1), m.group(2).strip()))
            else:
                translated_entries.append(("", ln))

    if not translated_entries:
        raise ValueError("Gemini no devolvió texto traducido.")

    return translated_entries, f"Gemini AI ({used_model})"


def translate_with_fallback(raw_entries, progress=None):
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


# ---------------------------------------------------------------------------
# PESTAÑA 2: FLUJO COMPLETO DESDE AUDIO
# ---------------------------------------------------------------------------


def transcribe_pipeline(
    file_path,
    title,
    audio_lang,
    target_mode,
    model_name,
    run_reconstruction,
    context_text,
    context_files,
    llm_model_choice,
    gemini_key_ui,
    progress=gr.Progress(),
):
    if not file_path:
        raise gr.Error("Primero sube un archivo de audio.")

    title = (title or "").strip() or "Transcripción de seminario"
    source_name = os.path.basename(file_path)
    stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(source_name)[0]).strip("_") or "transcripcion"

    progress(0.05, desc=f"Cargando modelo Whisper ({model_name})...")
    model = get_model(model_name)

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

    raw_entries = [
        (fmt_time(seg["start"]), seg["text"].strip())
        for seg in result["segments"]
        if seg["text"].strip()
    ]

    out_dir = tempfile.mkdtemp()
    orig_txt_path = os.path.join(out_dir, f"{stem}_original_{detected_lang}.txt")
    with open(orig_txt_path, "w", encoding="utf-8") as f:
        for tag, txt in raw_entries:
            f.write(f"[{tag}] {txt}\n")

    final_entries = raw_entries
    engine_used = None
    translated_to_spanish = False

    if target_mode == "to_es":
        if detected_lang == "es":
            engine_used = "El audio ya está en español."
        else:
            api_key = (gemini_key_ui or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
            if api_key:
                try:
                    final_entries, engine_used = translate_with_gemini(raw_entries, api_key, progress)
                    translated_to_spanish = True
                except Exception as e:
                    print(f"Aviso con Gemini ({e}); utilizando traductor alternativo...")
                    final_entries, engine_used = translate_with_fallback(raw_entries, progress)
                    engine_used += " (Fallback alternativo)"
                    translated_to_spanish = True
            else:
                final_entries, engine_used = translate_with_fallback(raw_entries, progress)
                translated_to_spanish = True

    # PDF de transcripción
    progress(0.85, desc="Generando PDF de transcripción...")
    suffix = "_espanol" if translated_to_spanish else ""
    transcript_pdf_path = os.path.join(out_dir, f"{stem}{suffix}.pdf")
    transcript_txt_path = os.path.join(out_dir, f"{stem}{suffix}.txt")

    full_transcript_str = "\n".join(f"[{tag}] {txt}" for tag, txt in final_entries)
    with open(transcript_txt_path, "w", encoding="utf-8") as f:
        f.write(full_transcript_str)

    if translated_to_spanish:
        doc_subtitle = (
            f"Archivo: {source_name} · Idioma original: {detected_lang.upper()} · "
            f"Traducido con {engine_used} · {datetime.now():%d/%m/%Y %H:%M}"
        )
    elif target_mode == "to_en":
        doc_subtitle = f"Archivo: {source_name} · Traducido a inglés con Whisper · {datetime.now():%d/%m/%Y %H:%M}"
    else:
        doc_subtitle = f"Archivo: {source_name} · Transcrito con Whisper ({detected_lang.upper()}) · {datetime.now():%d/%m/%Y %H:%M}"

    build_transcript_pdf(final_entries, title, doc_subtitle, transcript_pdf_path)

    # Reconstrucción opcional con LLM
    analysis_pdf_path = None
    analysis_md_path = None
    reconstruction_text = ""

    if run_reconstruction:
        progress(0.9, desc="Ejecutando reconstrucción y análisis con IA...")
        try:
            reconstruction_text, llm_engine = call_llm_for_reconstruction(
                transcription_text=full_transcript_str,
                context_text=context_text,
                context_files=context_files,
                title=title,
                model_choice=llm_model_choice,
                api_key_override=gemini_key_ui,
                progress=progress,
            )
            analysis_pdf_path = os.path.join(out_dir, f"{stem}_informe_reconstruido.pdf")
            analysis_md_path = os.path.join(out_dir, f"{stem}_informe_reconstruido.md")

            with open(analysis_md_path, "w", encoding="utf-8") as f:
                f.write(reconstruction_text)

            build_analysis_pdf(reconstruction_text, title, analysis_pdf_path)
        except Exception as e:
            reconstruction_text = f"⚠️ No se pudo completar la reconstrucción automática con IA: {e}"

    total_elapsed = time.time() - start_time
    status = (
        f"✅ **Transcripción completada en {total_elapsed / 60:.1f} min** (Whisper GPU: {whisper_elapsed / 60:.1f} min).\n"
        f"- Idioma detectado: **{detected_lang.upper()}**\n"
    )
    if translated_to_spanish:
        status += f"- Traducción al español: **{engine_used}**\n"
    if analysis_pdf_path:
        status += "- **Informe de reconstrucción y resumen ejecutivo generado con éxito.**\n"

    status += "📄 Descarga tus archivos abajo."
    preview_snippet = reconstruction_text if reconstruction_text else full_transcript_str[:3000]

    return (
        transcript_pdf_path,
        transcript_txt_path,
        orig_txt_path,
        analysis_pdf_path,
        analysis_md_path,
        preview_snippet,
        status,
    )


# ---------------------------------------------------------------------------
# INTERFAZ GRÁFICA (GRADIO)
# ---------------------------------------------------------------------------


def build_ui():
    custom_css = """
    .gradio-container { max-width: 1200px !important; }
    """
    with gr.Blocks(title="Transcriptor y Analizador de Seminarios", css=custom_css) as demo:
        gr.Markdown(
            "# 🎙️ Transcriptor y Analizador Académico de Seminarios\n"
            "Herramienta integral para **transcribir audios**, **reconstruir eventos con inicio tardío o cortes**, "
            "**generar resúmenes ejecutivos** y **rescatar preguntas y respuestas (Q&A)** con IA."
        )

        with gr.Tabs():
            # ===============================================================
            # PESTAÑA 1: POST-PROCESAR TRANSCRIPCIÓN PREVIA
            # ===============================================================
            with gr.Tab("📑 Post-procesar Transcripción Previa (PDF o TXT)"):
                gr.Markdown(
                    "### Reconstrucción y Resumen desde una transcripción existente\n"
                    "Sube el **PDF o TXT que ya generaste** (sin necesidad de volver a procesar el audio). "
                    "La IA reconstruirá el hilo conductor del seminario, subsanará inicios incompletos o cortes, "
                    "generará un resumen ejecutivo y analizará la sesión de preguntas."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        p1_file = gr.File(
                            label="📄 Transcripción previa (PDF o TXT)",
                            file_types=[".pdf", ".txt", ".md"],
                            type="filepath",
                        )
                        p1_title = gr.Textbox(
                            label="Título o tema del seminario",
                            value="Seminario de Investigación",
                            placeholder="Ej: Procesos Electroquímicos aplicados al tratamiento de aguas",
                        )
                        p1_context_text = gr.Textbox(
                            label="📧 Contexto adicional: Correo de anuncio, abstract o notas previas",
                            lines=4,
                            placeholder="Pega aquí el correo donde anunciaron el seminario, el nombre y laboratorio del ponente, el abstract o el temario...",
                        )
                        p1_context_files = gr.File(
                            label="📚 Archivos de apoyo opcionales (PDFs de artículos, diapositivas, capítulos)",
                            file_count="multiple",
                            file_types=[".pdf", ".txt"],
                            type="filepath",
                        )

                        p1_model = gr.Radio(
                            [
                                ("⚡ Groq Cloud (Llama 3.3 70B de Meta - Recomendado: 128k contexto, sin límites)", "Groq"),
                                ("🧠 Modelo Local en GPU Colab (Qwen 2.5 3B - Sin internet ni claves)", "Local"),
                                ("🌐 Google Gemini 3.8 Flash (con reintento)", "Gemini"),
                            ],
                            value="Groq",
                            label="Motor de Inteligencia Artificial para el Informe",
                            info="Groq con Llama 3.3 70B procesa seminarios completos en 3 segundos usando tu clave de Colab Secrets.",
                        )
                        p1_api_key = gr.Textbox(
                            label="API Key (Opcional: SOLO si eliges Groq o Gemini)",
                            type="password",
                            placeholder="NO necesaria para el Modelo Local en GPU",
                        )

                        p1_btn = gr.Button(
                            "🚀 Reconstruir Seminario y Generar Informe",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=5):
                        p1_status = gr.Markdown()
                        with gr.Row():
                            p1_pdf_out = gr.File(label="📄 Descargar Informe Reconstruido (PDF)")
                            p1_md_out = gr.File(label="📝 Descargar Informe en Markdown (.md)")
                        p1_preview = gr.Markdown(label="Vista previa del informe")

                p1_btn.click(
                    run_reconstruction_from_file,
                    inputs=[
                        p1_file,
                        p1_context_text,
                        p1_context_files,
                        p1_title,
                        p1_model,
                        p1_api_key,
                    ],
                    outputs=[p1_pdf_out, p1_md_out, p1_preview, p1_status],
                )

            # ===============================================================
            # PESTAÑA 2: FLUJO COMPLETO DESDE AUDIO
            # ===============================================================
            with gr.Tab("🎙️ Flujo Completo: Audio -> Transcripción + Reconstrucción"):
                gr.Markdown(
                    "### Transcripción de Audio en GPU T4 con Análisis Posterior Opcional\n"
                    "Sube un archivo de audio (en inglés, español u otro idioma). "
                    "Whisper transcribirá el audio en GPU, y si lo deseas, generará de inmediato el informe reconstruido."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        p2_audio = gr.File(label="Archivo de audio", file_types=AUDIO_TYPES, type="filepath")
                        p2_title = gr.Textbox(label="Título del documento", value="Transcripción de seminario")

                        p2_target_mode = gr.Radio(
                            [
                                ("Traducir al español 🇲🇽/🇪🇸 (Ideal para audios en inglés)", "to_es"),
                                ("Mantener idioma original (sin traducción)", "original"),
                                ("Traducir a inglés 🇺🇸 (Whisper nativo)", "to_en"),
                            ],
                            value="to_es",
                            label="Idioma de la transcripción",
                        )

                        p2_audio_lang = gr.Dropdown(
                            [
                                ("Detectar automáticamente", "auto"),
                                ("English (Inglés)", "en"),
                                ("Español", "es"),
                                ("Français", "fr"),
                                ("Deutsch", "de"),
                            ],
                            value="auto",
                            label="Idioma hablado en el audio",
                        )

                        p2_reconstruct_check = gr.Checkbox(
                            label="Generar también Reconstrucción e Informe Ejecutivo con IA",
                            value=True,
                            info="Combina la transcripción con el contexto para llenar vacíos, crear resumen y analizar Q&A.",
                        )

                        p2_context_text = gr.Textbox(
                            label="📧 Contexto adicional (Correo de anuncio, abstract o notas)",
                            lines=3,
                            placeholder="Pega aquí el correo de invitación, abstract o tema...",
                        )
                        p2_context_files = gr.File(
                            label="📚 Archivos de apoyo (PDFs de artículos, diapositivas)",
                            file_count="multiple",
                            file_types=[".pdf", ".txt"],
                            type="filepath",
                        )

                        with gr.Accordion("⚙️ Opciones avanzadas de Whisper e IA", open=False):
                            p2_whisper_model = gr.Dropdown(
                                ["small", "medium", "large-v3"],
                                value="medium",
                                label="Modelo Whisper",
                            )
                            p2_llm_model = gr.Radio(
                                [
                                    ("⚡ Groq Cloud (Llama 3.3 70B de Meta - Recomendado)", "Groq"),
                                    ("🧠 Modelo Local en GPU Colab (Qwen 2.5 3B)", "Local"),
                                    ("🌐 Google Gemini 3.8 Flash", "Gemini"),
                                ],
                                value="Groq",
                                label="Modelo para Reconstrucción",
                            )
                            p2_api_key = gr.Textbox(
                                label="API Key de Gemini / Groq (opcional)",
                                type="password",
                            )

                        p2_btn = gr.Button("Transcribir y Procesar", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        p2_status = gr.Markdown()
                        gr.Markdown("#### Archivos de Transcripción:")
                        with gr.Row():
                            p2_pdf_trans = gr.File(label="📄 PDF Transcripción (Línea por línea)")
                            p2_txt_trans = gr.File(label="📝 TXT Transcripción")
                        p2_orig_txt = gr.File(label="🗂️ TXT Original en inglés (Respaldo)")

                        gr.Markdown("#### Informe Ejecutivo Reconstruido:")
                        with gr.Row():
                            p2_pdf_analysis = gr.File(label="📄 PDF Informe Reconstruido")
                            p2_md_analysis = gr.File(label="📝 Markdown Informe Reconstruido")

                        p2_preview = gr.Textbox(label="Vista previa del texto", lines=10, interactive=False)

                p2_btn.click(
                    transcribe_pipeline,
                    inputs=[
                        p2_audio,
                        p2_title,
                        p2_audio_lang,
                        p2_target_mode,
                        p2_whisper_model,
                        p2_reconstruct_check,
                        p2_context_text,
                        p2_context_files,
                        p2_llm_model,
                        p2_api_key,
                    ],
                    outputs=[
                        p2_pdf_trans,
                        p2_txt_trans,
                        p2_orig_txt,
                        p2_pdf_analysis,
                        p2_md_analysis,
                        p2_preview,
                        p2_status,
                    ],
                )

    return demo


if __name__ == "__main__":
    if DEVICE == "cpu":
        print("ℹ️ Ejecutando en CPU. Para transcribir audio rápido activa la GPU T4 en Colab.")
    else:
        print(f"✅ Dispositivo: CUDA ({torch.cuda.get_device_name(0)})")

    user = os.environ.get("APP_USER", "seminarios")
    password = os.environ.get("APP_PASSWORD", "")

    build_ui().queue(default_concurrency_limit=1).launch(
        share=True,
        auth=(user, password) if password else None,
    )
