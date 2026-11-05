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
import shutil
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


def get_event_storage_dirs(title):
    """Devuelve el directorio local outputs/<nombre_evento> y, si Google Drive está montado en Colab,
    la ruta a /content/drive/MyDrive/transcripciones/<nombre_evento>."""
    raw_title = (title or "").strip()
    clean_title = re.sub(r'[\\/*?:"<>|]', "", raw_title).strip().replace(" ", "_")
    clean_title = re.sub(r"_+", "_", clean_title).strip("_")

    if not clean_title or clean_title.lower() in [
        "seminario", "clase", "conferencia", "transcripcion", "evento", "seminario_de_investigacion"
    ]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_title = f"{clean_title or 'evento'}_{ts}"

    local_dir = os.path.join(os.path.abspath("outputs"), clean_title)
    os.makedirs(local_dir, exist_ok=True)

    drive_base = "/content/drive/MyDrive"
    gdrive_dir = None
    if os.path.isdir(drive_base):
        target = os.path.join(drive_base, "transcripciones", clean_title)
        try:
            os.makedirs(target, exist_ok=True)
            gdrive_dir = target
        except Exception as e:
            print(f"Aviso al acceder a la carpeta de Google Drive: {e}")

    return local_dir, gdrive_dir, clean_title


def sync_to_drive(file_paths, gdrive_dir):
    """Copia los archivos generados a Google Drive en la subcarpeta del evento si está disponible."""
    if not gdrive_dir:
        return []
    copied = []
    for fp in file_paths:
        if fp and os.path.exists(fp):
            try:
                dest = os.path.join(gdrive_dir, os.path.basename(fp))
                shutil.copy2(fp, dest)
                copied.append(os.path.basename(fp))
            except Exception as e:
                print(f"Error copiando {fp} a Google Drive: {e}")
    return copied


def clean_transcript_text(text):
    """Limpia la transcripción eliminando encabezados de página de PDFs previos y
    colapsando bucles de repetición/alucinaciones de Whisper."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = []
    prev_content = None
    rep_count = 0

    for line in lines:
        l = line.strip()
        if not l:
            continue
        # Descartar encabezados y pies de página típicos de exportaciones previas
        if re.match(r"^P[aá]gina \d+", l, re.IGNORECASE):
            continue
        if "Transcripci" in l and "seminario" in l:
            continue
        if l.startswith("Archivo:") and "Idioma original:" in l:
            continue
        if "Informe T" in l and "Transcrip AI" in l:
            continue
        if "Informe Ejecutivo y Reconstrucci" in l:
            continue

        # Extraer contenido puro sin marcas de tiempo [HH:MM:SS] para detectar bucles
        content_only = re.sub(r"^\[\d{1,2}:\d{2}:\d{2}\]\s*|^\[\d{1,2}:\d{2}\]\s*", "", l).strip()
        if not content_only:
            continue

        if prev_content and content_only.lower() == prev_content.lower():
            rep_count += 1
            if rep_count == 1:
                # Permitir máximo 1 repetición legítima
                cleaned.append(l)
            elif rep_count == 2:
                cleaned.append("[... repeticiones idénticas omitidas ...]")
            # Más de 2 repeticiones sucesivas son descartadas
        else:
            rep_count = 0
            prev_content = content_only
            cleaned.append(l)

    return "\n".join(cleaned)


def safe_truncate_transcript(text, max_chars=45000):
    """Asegura que el texto de la transcripción quepa en el contexto del modelo,
    conservando el inicio (60%) y el desenlace/Q&A (40%), nunca truncando abruptamente."""
    if not text or len(text) <= max_chars:
        return text
    part1 = int(max_chars * 0.6)
    part2 = int(max_chars * 0.4)
    return (
        text[:part1]
        + "\n\n[... PARTE CENTRAL CONDENSADA POR EXTENSIÓN PARA PRESERVAR INICIO Y CONCLUSIONES/PREGUNTAS ...]\n\n"
        + text[-part2:]
    )


def extract_text_from_file(file_path):
    """Extrae texto de un archivo PDF o TXT y aplica limpieza de bucles y encabezados."""
    if not file_path:
        return ""
    real_path = file_path if isinstance(file_path, str) else getattr(file_path, "name", str(file_path))
    if not os.path.exists(real_path):
        return ""
    ext = os.path.splitext(real_path)[1].lower()
    raw_text = ""
    if ext == ".txt" or ext == ".md":
        try:
            with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
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
            raw_text = "\n\n".join(pages)
        except Exception as e:
            return f"[Error extrayendo texto del PDF: {e}]"
    
    return clean_transcript_text(raw_text)


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

CLASS_NOTES_PROMPT = """Eres un catedrático universitario y pedagogo experto en la elaboración de apuntes académicos de élite y guías de estudio avanzadas.
Tu objetivo es transformar la transcripción de una clase universitaria, cátedra o curso especializado en un CUADERNO DE APUNTES Y GUÍA DE ESTUDIO estructurado, claro, riguroso y exhaustivo.
Si la grabación comenzó tarde o se cortó, reconstruye los conceptos iniciales a partir del contexto provisto (temario, diapositivas, notas del curso) indicándolo explícitamente con *[Reconstrucción contextual: ...]*.

Debes estructurar tus apuntes en Markdown con las siguientes 5 secciones principales (utiliza encabezados '##'):

## 1. Ficha de la Clase y Objetivos de Aprendizaje
- **Materia o Curso:** Asignatura o disciplina correspondiente.
- **Tema Central de la Sesión:** Título temático formal de la clase.
- **Profesor / Catedrático:** Nombre del docente si se menciona o deduce.
- **Objetivos de Aprendizaje:** 3 a 5 competencias o conocimientos clave que el estudiante debe dominar tras esta clase.
- **Diagnóstico del registro:** Estado de la grabación y secciones reconstruidas si las hubo.

## 2. Cuaderno de Apuntes Estructurado (Desarrollo Pedagógico de la Clase)
- Desarrolla todos los temas vistos en clase en un orden didáctico impecable.
- **Fundamentos y Teoría:** Explicación detallada y sin rodeos de cada concepto, ley, principio físico/químico/matemático o modelo teórico expuesto.
- **Fórmulas, Ecuaciones y Notación:** Desglosa con precisión las expresiones matemáticas, definiendo cada variable, constante y unidades de medida.
- **Ejemplos y Casos de Estudio:** Explica paso a paso los ejercicios, demostraciones, gráficas o aplicaciones prácticas presentadas por el profesor.
- **Diagramas o Esquemas Conceptuales:** Describe textualmente la lógica de diagramas o experimentos abordados en clase.

## 3. "Lo que el Profesor Enfatizó" (Atención para Exámenes)
- **Advertencias y Errores Comunes:** Puntos donde los alumnos suelen equivocarse y que el docente remarcó.
- **Preguntas típicas de evaluación:** Conceptos clave con alta probabilidad de figurar en un examen parcial o final.
- **Criterios de rigor:** Condiciones de validez o límites de aplicabilidad de las teorías vistas.

## 4. Glosario de Términos y Fórmulas Clave
- Lista de 8 a 12 términos técnicos especializados introducidos en la sesión con su definición concisa.
- Cuadro síntesis de fórmulas esenciales vistas en la clase.

## 5. Cuestionario de Repaso y Autoevaluación
- Banco de 5 a 8 preguntas conceptuales y de aplicación basadas estrictamente en la clase.
- **Para cada pregunta:** Proporciona la solución o respuesta modelo detallada para que el estudiante pueda autoevaluarse antes del examen.

IMPORTANTE: Escribe en un español académico didáctico, claro y riguroso; inicia directamente con el encabezado '## 1. Ficha de la Clase y Objetivos de Aprendizaje'.
"""

SEMINAR_PROMPT = """Eres un investigador y analista académico experto en síntesis y reconstrucción de seminarios de investigación y conferencias científicas.
Tu objetivo es procesar la transcripción y el contexto disponible (correos de anuncio, abstract, diapositivas, artículos) para generar un informe técnico de altísimo nivel, subsanando inicios tardíos, cortes de audio o baches acústicos.

Debes estructurar tu respuesta en Markdown con las siguientes 4 secciones principales (utiliza encabezados '##'):

## 1. Ficha Técnica y Diagnóstico de la Grabación
- **Título formal del seminario:** Deduce o valida el título más adecuado.
- **Ponente / Expositor:** Nombre, grado académico e institución/laboratorio de procedencia.
- **Área temática y Palabras clave:** 5 a 8 términos clave especializados.
- **Diagnóstico del registro:** Diagnóstico explícito del estado de la grabación (ej: 'Grabación completa', 'Grabación con inicio tardío - faltó la introducción original y fue reconstruida con base en el anuncio', 'Cortes intermedios').

## 2. Reconstrucción Integral del Seminario (Hilo Conductor)
- Reconstruye la totalidad de la exposición en orden cronológico, fluido y lógico.
- **Si el audio empezó a mitad de camino:** utiliza el correo de anuncio, el título y el desarrollo de la charla para reconstruir la introducción ausente, el planteamiento del problema, los antecedentes y la motivación, señalándolo explícitamente con la etiqueta *[Reconstrucción contextual a partir de anuncio/referencias: ...]* para distinguir lo deducido de lo grabado.
- Desarrolla a profundidad los temas expuestos: motivación, metodología experimental, técnicas analíticas, resultados principales, discusión de datos, gráficas explicadas y conclusiones.
- Incluye términos técnicos rigurosos, nombres de compuestos, valores numéricos y relaciones causa-efecto.

## 3. Resumen Ejecutivo y Puntos Rescatables ('Key Takeaways')
- **Resumen ejecutivo:** Síntesis clara y fluida de 3 a 4 párrafos de alto impacto.
- **Key Takeaways (Puntos rescatables):** Lista de 6 a 10 ideas o conclusiones fundamentales.
- **Hallazgos sorprendentes o advertencias metodológicas:** Aspectos críticos, limitaciones o recomendaciones destacadas por el ponente.

## 4. Sesión de Preguntas y Respuestas (Q&A) y Debate
- Desglosa cada pregunta formulada por el público/moderadores y sintetiza la respuesta del ponente.
- Si no hubo sesión de preguntas grabada, añade *"Preguntas críticas sugeridas para el trabajo presentado"* (3-4 preguntas con sus respuestas o hipótesis fundamentadas).

IMPORTANTE: Escribe en un español académico impecable, formal y sin preámbulos conversacionales; inicia directamente con '## 1. Ficha Técnica y Diagnóstico de la Grabación'.
"""

HYBRID_PROMPT = """Eres un investigador y catedrático universitario de élite. Tu objetivo es generar un DOCUMENTO MAESTRO INTEGRAL que combine un Informe Técnico Académico con un Cuaderno Completo de Apuntes de Estudio a partir de la transcripción de una conferencia, clase avanzada o seminario.

Debes estructurar tu respuesta en Markdown con las siguientes 5 secciones principales (utiliza encabezados '##'):

## 1. Ficha Técnica y Objetivos de la Sesión
- **Título formal del evento / clase:**
- **Expositor / Catedrático:**
- **Área del conocimiento y Palabras clave:**
- **Diagnóstico del registro de audio y reconstrucciones contextuales:**

## 2. Reconstrucción y Apuntes Detallados de la Sesión
- Reconstrucción cronológica y pedagógica del tema (incluyendo introducción ausente si hubo corte al inicio).
- Teoría, derivaciones, relaciones causa-efecto y principios científicos.
- Fórmulas, ecuaciones, parámetros adimensionales y unidades.
- Discusión de datos experimentales, gráficas y metodología.

## 3. Resumen Ejecutivo y Puntos Clave ('Key Takeaways')
- Resumen ejecutivo de alto nivel (3 párrafos).
- 8 puntos clave indispensables.
- Advertencias metodológicas o hallazgos críticos.

## 4. "Atención para Exámenes e Investigación" y Glosario
- Puntos críticos destacados por el ponente/profesor.
- Glosario de términos y fórmulas clave.

## 5. Sesión de Preguntas (Q&A) y Cuestionario de Autoevaluación
- Desglose de preguntas del público o 5 preguntas clave de evaluación con sus respuestas explicadas.

IMPORTANTE: Escribe en español académico impecable y formal; inicia directamente con '## 1. Ficha Técnica y Objetivos de la Sesión'.
"""


def call_llm_for_reconstruction(
    transcription_text,
    context_text,
    context_files,
    title,
    model_choice,
    doc_type="class_notes",
    api_key_override=None,
    progress=None,
):
    """Envía la transcripción y el contexto al LLM para generar apuntes o informe reconstructivo."""
    # 1. Asegurar limpieza y formato compacto de la transcripción
    cleaned_transcript = clean_transcript_text(transcription_text)

    # 2. Armar contexto adicional (abstracts, correos, notas, PDFs)
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
                    context_blocks.append(f"=== ARCHIVO DE APOYO ({bname}) ===\n{txt[:25000]}")

    full_context_str = "\n\n".join(context_blocks) if context_blocks else "[No se proporcionó contexto adicional]"

    # 3. Truncado seguro preservando inicio y desenlace si la transcripción es inusualmente kilométrica
    safe_transcript = safe_truncate_transcript(cleaned_transcript, max_chars=42000)

    # 4. Seleccionar plantilla según el tipo de documento solicitado
    if doc_type == "class_notes":
        system_prompt = CLASS_NOTES_PROMPT
        doc_label = "Cuaderno de Apuntes de Clase"
        final_instruction = (
            "Con base en la transcripción y el contexto, redacta de inmediato el CUADERNO DE APUNTES Y GUÍA DE ESTUDIO "
            "estructurado en Markdown con sus 5 secciones principales (## 1. Ficha de la Clase y Objetivos de Aprendizaje, "
            "## 2. Cuaderno de Apuntes Estructurado, ## 3. 'Lo que el Profesor Enfatizó', ## 4. Glosario de Términos y Fórmulas Clave, "
            "## 5. Cuestionario de Repaso y Autoevaluación). Redacta en español pedagógico y riguroso e inicia directamente con el "
            "encabezado '## 1. Ficha de la Clase y Objetivos de Aprendizaje'."
        )
    elif doc_type == "hybrid":
        system_prompt = HYBRID_PROMPT
        doc_label = "Documento Maestro (Informe + Apuntes)"
        final_instruction = (
            "Con base en la transcripción y el contexto, genera el DOCUMENTO MAESTRO INTEGRAL en Markdown con sus 5 secciones "
            "(## 1. Ficha Técnica y Objetivos de la Sesión, ## 2. Reconstrucción y Apuntes Detallados, ## 3. Resumen Ejecutivo y Puntos Clave, "
            "## 4. 'Atención para Exámenes e Investigación' y Glosario, ## 5. Sesión de Preguntas (Q&A) y Cuestionario de Autoevaluación). "
            "Inicia directamente con '## 1. Ficha Técnica y Objetivos de la Sesión'."
        )
    else:
        system_prompt = SEMINAR_PROMPT
        doc_label = "Informe de Seminario"
        final_instruction = (
            "Con base en la transcripción y el contexto, genera el INFORME TÉCNICO DE SEMINARIO en Markdown con sus 4 secciones "
            "(## 1. Ficha Técnica y Diagnóstico de la Grabación, ## 2. Reconstrucción Integral del Seminario, ## 3. Resumen Ejecutivo y Puntos Rescatables, "
            "## 4. Sesión de Preguntas y Respuestas (Q&A) y Debate). Inicia directamente con '## 1. Ficha Técnica y Diagnóstico de la Grabación'."
        )

    user_prompt = (
        f"# TÍTULO SUGERIDO O TEMA: {title}\n\n"
        f"# INFORMACIÓN CONTEXTUAL DE APOYO (Abstract, temario o notas previas):\n{full_context_str}\n\n"
        f"# TRANSCRIPCIÓN REGISTRADA:\n{safe_transcript}\n\n"
        f"# INSTRUCCIÓN FINAL:\n{final_instruction}"
    )

    if "Local" in model_choice or "Qwen" in model_choice:
        if progress:
            progress(0.15, desc=f"Cargando modelo local Qwen 2.5 3B para generar {doc_label}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Soporte nativo de bfloat16 en GPUs modernas (A100 / L4) o float16 en T4
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
        attn_impl = "sdpa" if torch.cuda.is_available() else "eager"

        model_id = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation=attn_impl,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        prompt_len = model_inputs.input_ids.shape[1]

        if progress:
            progress(0.40, desc=f"Generando {doc_label} en GPU ({prompt_len} tokens de entrada)...")

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=3500,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extraer estrictamente los nuevos tokens generados por el asistente
        response_tokens = generated_ids[0][prompt_len:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Local"
        return response_text, f"Qwen 2.5 3B (Local en {dev_name})"

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
        groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        last_err = None

        for groq_model in groq_models:
            try:
                if progress:
                    progress(0.4, desc=f"Generando {doc_label} con {groq_model} en Groq...")
                response = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=3500,
                )
                content = response.choices[0].message.content or ""
                if content.strip():
                    return content.strip(), f"Meta ({groq_model} vía Groq)"
            except Exception as e:
                last_err = e
                print(f"Aviso Groq {groq_model}: {e}")
                continue
        raise gr.Error(f"No se pudo completar con Groq ({last_err}). Te sugerimos seleccionar 'Modelo Local en GPU Colab' para correr directamente en tu GPU sin cuotas de API.")

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
                    contents=[system_prompt + "\n\n" + user_prompt],
                )
                if response.text and response.text.strip():
                    return response.text.strip(), "Google Gemini 3.8 Flash"
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
                    wait_sec = 4 * (attempt + 1)
                    print(f"Gemini 3.8 saturado (503). Reintentando en {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                else:
                    raise e
        raise gr.Error("Google Gemini 3.8 sigue saturado por alta demanda (503). Por favor selecciona 'Modelo Local en GPU Colab' o 'Groq Cloud'.")

    else:
        raise gr.Error(f"Modelo no soportado: {model_choice}")


def run_reconstruction_from_file(
    file_path,
    context_text,
    context_files,
    title,
    doc_type,
    model_choice,
    api_key_override,
    progress=gr.Progress(),
):
    """Pestaña 2: Procesa una transcripción ya existente (PDF o TXT)."""
    if not file_path:
        raise gr.Error("Primero sube un archivo PDF o TXT con la transcripción previa.")

    title = (title or "").strip() or "Clase o Seminario"
    file_real_path = file_path if isinstance(file_path, str) else getattr(file_path, "name", str(file_path))
    source_name = os.path.basename(file_real_path)

    local_dir, gdrive_dir, event_slug = get_event_storage_dirs(title)

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
        doc_type=doc_type,
        api_key_override=api_key_override,
        progress=progress,
    )
    elapsed = time.time() - start_time

    progress(0.85, desc="Generando PDF del documento académico...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_suffix = "apuntes" if doc_type == "class_notes" else ("integral" if doc_type == "hybrid" else "informe")
    out_pdf_path = os.path.join(local_dir, f"{event_slug}_{doc_suffix}_{ts}.pdf")
    out_md_path = os.path.join(local_dir, f"{event_slug}_{doc_suffix}_{ts}.md")

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_result)

    build_analysis_pdf(md_result, title, out_pdf_path)

    copied = sync_to_drive([out_pdf_path, out_md_path], gdrive_dir)
    drive_info = ""
    if copied:
        drive_info = f"\n- ☁️ **Guardado automáticamente en Google Drive:** Carpeta `Mi unidad/transcripciones/{event_slug}/` ({', '.join(copied)})"

    word_count = len(transcript_text.split())
    status_msg = (
        f"✅ **Documento generado con éxito en {elapsed:.1f} segundos**.\n\n"
        f"- **Motor de IA utilizado:** {engine_name}\n"
        f"- **Archivo analizado:** `{source_name}` (~{word_count} palabras)\n"
        f"- **Carpeta del evento:** `{event_slug}`"
        f"{drive_info}\n\n"
        "📄 **Descarga tu PDF y archivo Markdown abajo**, o consúltalos directamente en pantalla o en tu Google Drive."
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

        candidates = ["gemini-3.8-flash"]
        response = None
        used_model = "gemini-3.8-flash"
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
# FLUJO COMPLETO DESDE AUDIO (PESTAÑA 1)
# ---------------------------------------------------------------------------


def transcribe_pipeline(
    file_path,
    title,
    doc_type,
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

    title = (title or "").strip() or "Clase o Seminario"
    source_name = os.path.basename(file_path)

    local_dir, gdrive_dir, event_slug = get_event_storage_dirs(title)

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
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
    )
    whisper_elapsed = time.time() - start_time
    detected_lang = result.get("language", audio_lang)

    raw_entries = [
        (fmt_time(seg["start"]), seg["text"].strip())
        for seg in result["segments"]
        if seg["text"].strip()
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_txt_path = os.path.join(local_dir, f"{event_slug}_{ts}_original_{detected_lang}.txt")
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
    suffix_lang = "_espanol" if translated_to_spanish else ""
    transcript_pdf_path = os.path.join(local_dir, f"{event_slug}_{ts}_transcripcion{suffix_lang}.pdf")
    transcript_txt_path = os.path.join(local_dir, f"{event_slug}_{ts}_transcripcion{suffix_lang}.txt")

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

    # Reconstrucción / Apuntes opcional con LLM
    analysis_pdf_path = None
    analysis_md_path = None
    reconstruction_text = ""

    if run_reconstruction:
        progress(0.9, desc="Generando apuntes / informe con IA...")
        try:
            reconstruction_text, llm_engine = call_llm_for_reconstruction(
                transcription_text=full_transcript_str,
                context_text=context_text,
                context_files=context_files,
                title=title,
                model_choice=llm_model_choice,
                doc_type=doc_type,
                api_key_override=gemini_key_ui,
                progress=progress,
            )
            doc_suffix = "apuntes" if doc_type == "class_notes" else ("integral" if doc_type == "hybrid" else "informe")
            analysis_pdf_path = os.path.join(local_dir, f"{event_slug}_{ts}_{doc_suffix}.pdf")
            analysis_md_path = os.path.join(local_dir, f"{event_slug}_{ts}_{doc_suffix}.md")

            with open(analysis_md_path, "w", encoding="utf-8") as f:
                f.write(reconstruction_text)

            build_analysis_pdf(reconstruction_text, title, analysis_pdf_path)
        except Exception as e:
            reconstruction_text = f"⚠️ No se pudo completar la generación con IA: {e}"

    # Sincronizar todos los archivos a Google Drive
    files_to_sync = [transcript_pdf_path, transcript_txt_path, orig_txt_path]
    if analysis_pdf_path and os.path.exists(analysis_pdf_path):
        files_to_sync.extend([analysis_pdf_path, analysis_md_path])

    copied = sync_to_drive(files_to_sync, gdrive_dir)
    drive_info = ""
    if copied:
        drive_info = f"\n\n☁️ **Guardado automáticamente en Google Drive:**\n- Carpeta: `Mi unidad/transcripciones/{event_slug}/`\n- Archivos respaldados: {len(copied)} ({', '.join(copied)})"

    total_elapsed = time.time() - start_time
    status = (
        f"✅ **Transcripción completada en {total_elapsed / 60:.1f} min** (Whisper GPU: {whisper_elapsed / 60:.1f} min).\n"
        f"- Idioma detectado: **{detected_lang.upper()}**\n"
    )
    if translated_to_spanish:
        status += f"- Traducción al español: **{engine_used}**\n"
    if analysis_pdf_path:
        status += "- **Documento de apuntes / informe generado con éxito.**\n"
    status += f"- **Carpeta del evento:** `{event_slug}`"
    if drive_info:
        status += drive_info

    status += "\n\n📄 Descarga tus archivos abajo o consúltalos en tu Google Drive."
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
    with gr.Blocks(title="Transcriptor y Generador Académico", css=custom_css) as demo:
        gr.Markdown(
            "# 🎙️ Transcriptor y Generador Académico de Apuntes e Informes\n"
            "Herramienta integral para **transcribir audios**, **elaborar apuntes de clase y guías de estudio**, "
            "**reconstruir seminarios con inicio tardío o cortes** y **sincronizar todo en Google Drive**."
        )

        with gr.Tabs():
            # ===============================================================
            # PESTAÑA 1: FLUJO COMPLETO DESDE AUDIO
            # ===============================================================
            with gr.Tab("🎙️ Flujo Completo: Audio -> Transcripción + Apuntes / Informe"):
                gr.Markdown(
                    "### Transcripción en GPU con Generación Automática de Apuntes o Informe\n"
                    "Sube el archivo de audio de tu clase, seminario o conferencia. Whisper transcribirá y traducirá "
                    "el audio con GPU, y la IA estructurará de inmediato tus apuntes de estudio o informe ejecutivo en una carpeta propia en Drive."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        p1_audio = gr.File(label="🎵 Archivo de audio", file_types=AUDIO_TYPES, type="filepath")
                        p1_title = gr.Textbox(
                            label="Nombre o tema de la clase / seminario (Da nombre a la carpeta)",
                            value="Clase de Investigación",
                            placeholder="Ej: Termodinámica Estadística - Clase 4",
                        )
                        p1_doc_type = gr.Radio(
                            [
                                ("📖 Cuaderno de Apuntes de Clase (Fórmulas, conceptos, lo que el profesor enfatizó, glosario y autoevaluación)", "class_notes"),
                                ("🎓 Informe Ejecutivo de Seminario (Ficha técnica, reconstrucción cronológica, resumen y Q&A)", "seminar"),
                                ("🌟 Híbrido Integral (Ambos combinados: informe + apuntes de estudio completos)", "hybrid"),
                            ],
                            value="class_notes",
                            label="Tipo de documento académico a generar",
                            info="Selecciona 'Apuntes de Clase' para explicaciones didácticas paso a paso, fórmulas desglosadas y cuestionario tipo examen.",
                        )

                        p1_target_mode = gr.Radio(
                            [
                                ("Traducir al español 🇲🇽/🇪🇸 (Ideal para clases o charlas en inglés)", "to_es"),
                                ("Mantener idioma original (sin traducción)", "original"),
                                ("Traducir a inglés 🇺🇸 (Whisper nativo)", "to_en"),
                            ],
                            value="to_es",
                            label="Idioma de la transcripción",
                        )

                        p1_audio_lang = gr.Dropdown(
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

                        p1_reconstruct_check = gr.Checkbox(
                            label="Generar Apuntes / Informe Académico con IA",
                            value=True,
                            info="Combina la transcripción con el contexto para llenar vacíos, explicar conceptos y armar la guía.",
                        )

                        p1_context_text = gr.Textbox(
                            label="📧 Contexto adicional (Temario, abstract, correo de anuncio o notas del curso)",
                            lines=3,
                            placeholder="Pega aquí el temario de la clase, correo de invitación, abstract o tema clave...",
                        )
                        p1_context_files = gr.File(
                            label="📚 Archivos de apoyo opcionales (Diapositivas, artículos, capítulos)",
                            file_count="multiple",
                            file_types=[".pdf", ".txt"],
                            type="filepath",
                        )

                        with gr.Accordion("⚙️ Opciones avanzadas de Whisper e IA", open=False):
                            p1_whisper_model = gr.Dropdown(
                                ["small", "medium", "large-v3"],
                                value="medium",
                                label="Modelo Whisper",
                            )
                            p1_llm_model = gr.Radio(
                                [
                                    ("🧠 Modelo Local en GPU Colab (Qwen 2.5 3B - Recomendado en A100/L4: 100% autónomo)", "Local"),
                                    ("⚡ Groq Cloud (Llama 3.3 70B)", "Groq"),
                                    ("🌐 Google Gemini 3.8 Flash", "Gemini"),
                                ],
                                value="Local",
                                label="Motor de IA para Apuntes / Informe",
                            )
                            p1_api_key = gr.Textbox(
                                label="API Key de Gemini / Groq (opcional)",
                                type="password",
                            )

                        p1_btn = gr.Button("🚀 Iniciar Transcripción y Generar Documentos", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        p1_status = gr.Markdown()
                        gr.Markdown("#### Archivos de Transcripción:")
                        with gr.Row():
                            p1_pdf_trans = gr.File(label="📄 PDF Transcripción (Línea por línea)")
                            p1_txt_trans = gr.File(label="📝 TXT Transcripción")
                        p1_orig_txt = gr.File(label="🗂️ TXT Original en inglés (Respaldo)")

                        gr.Markdown("#### Apuntes / Informe Académico Generado:")
                        with gr.Row():
                            p1_pdf_analysis = gr.File(label="📄 PDF Apuntes / Informe")
                            p1_md_analysis = gr.File(label="📝 Markdown (.md)")

                        p1_preview = gr.Textbox(label="Vista previa del texto", lines=10, interactive=False)

                p1_btn.click(
                    transcribe_pipeline,
                    inputs=[
                        p1_audio,
                        p1_title,
                        p1_doc_type,
                        p1_audio_lang,
                        p1_target_mode,
                        p1_whisper_model,
                        p1_reconstruct_check,
                        p1_context_text,
                        p1_context_files,
                        p1_llm_model,
                        p1_api_key,
                    ],
                    outputs=[
                        p1_pdf_trans,
                        p1_txt_trans,
                        p1_orig_txt,
                        p1_pdf_analysis,
                        p1_md_analysis,
                        p1_preview,
                        p1_status,
                    ],
                )

            # ===============================================================
            # PESTAÑA 2: POST-PROCESAR TRANSCRIPCIÓN PREVIA
            # ===============================================================
            with gr.Tab("📑 Post-procesar Transcripción Previa (PDF o TXT)"):
                gr.Markdown(
                    "### Generación de Apuntes e Informes desde una Transcripción Existente\n"
                    "Sube el **PDF o TXT que ya generaste** (sin necesidad de volver a procesar el audio). "
                    "La IA reconstruirá el contenido faltante o dañado por cortes, subsanará inicios incompletos y generará tus apuntes o informe en su propia carpeta."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        p2_file = gr.File(
                            label="📄 Transcripción previa (PDF o TXT)",
                            file_types=[".pdf", ".txt", ".md"],
                            type="filepath",
                        )
                        p2_title = gr.Textbox(
                            label="Nombre o tema de la clase / seminario (Da nombre a la carpeta)",
                            value="Seminario de Investigación",
                            placeholder="Ej: Procesos Electroquímicos aplicados al tratamiento de aguas",
                        )
                        p2_doc_type = gr.Radio(
                            [
                                ("📖 Cuaderno de Apuntes de Clase (Fórmulas, conceptos, advertencias de examen, glosario y autoevaluación)", "class_notes"),
                                ("🎓 Informe Ejecutivo de Seminario (Ficha técnica, reconstrucción cronológica, resumen y Q&A)", "seminar"),
                                ("🌟 Híbrido Integral (Ambos combinados: informe + apuntes de estudio completos)", "hybrid"),
                            ],
                            value="class_notes",
                            label="Tipo de documento académico a generar",
                        )
                        p2_context_text = gr.Textbox(
                            label="📧 Contexto adicional: Correo de anuncio, temario, abstract o notas previas",
                            lines=4,
                            placeholder="Pega aquí el temario, el correo donde anunciaron el evento, nombre del profesor/ponente, etc...",
                        )
                        p2_context_files = gr.File(
                            label="📚 Archivos de apoyo opcionales (Diapositivas, artículos, capítulos)",
                            file_count="multiple",
                            file_types=[".pdf", ".txt"],
                            type="filepath",
                        )

                        p2_model = gr.Radio(
                            [
                                ("🧠 Modelo Local en GPU Colab (Qwen 2.5 3B - Recomendado en A100/L4: 100% autónomo)", "Local"),
                                ("⚡ Groq Cloud (Llama 3.3 70B - Ultrarrápido, usa tu GROQ_API_KEY)", "Groq"),
                                ("🌐 Google Gemini 3.8 Flash (Requiere GEMINI_API_KEY)", "Gemini"),
                            ],
                            value="Local",
                            label="Motor de Inteligencia Artificial",
                            info="En Colab Pro con GPU A100 o L4, el Modelo Local analiza todo en ~15 segundos en tu GPU.",
                        )
                        p2_api_key = gr.Textbox(
                            label="API Key (Opcional: SOLO si eliges Groq o Gemini)",
                            type="password",
                            placeholder="NO necesaria para el Modelo Local en GPU",
                        )

                        p2_btn = gr.Button(
                            "🚀 Reconstruir Contenido y Generar Documentos",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=5):
                        p2_status = gr.Markdown()
                        with gr.Row():
                            p2_pdf_out = gr.File(label="📄 Descargar Apuntes / Informe (PDF)")
                            p2_md_out = gr.File(label="📝 Descargar en Markdown (.md)")
                        p2_preview = gr.Markdown(label="Vista previa del documento")

                p2_btn.click(
                    run_reconstruction_from_file,
                    inputs=[
                        p2_file,
                        p2_context_text,
                        p2_context_files,
                        p2_title,
                        p2_doc_type,
                        p2_model,
                        p2_api_key,
                    ],
                    outputs=[p2_pdf_out, p2_md_out, p2_preview, p2_status],
                )

    return demo


if __name__ == "__main__":
    if DEVICE == "cpu":
        print("ℹ️ Ejecutando en CPU. Para transcribir audio rápido activa la GPU (L4 o A100) en Colab.")
    else:
        print(f"✅ Dispositivo: CUDA ({torch.cuda.get_device_name(0)})")

    user = os.environ.get("APP_USER", "seminarios")
    password = os.environ.get("APP_PASSWORD", "")

    outputs_dir = os.path.abspath("outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    build_ui().queue(default_concurrency_limit=1).launch(
        share=True,
        auth=(user, password) if password else None,
        allowed_paths=[outputs_dir, tempfile.gettempdir()],
    )
