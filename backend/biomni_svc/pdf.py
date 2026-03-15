"""PDF report generation for Biomni runs and sessions."""
import html
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

PDF_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Biomni Report</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2em; line-height: 1.5; }}
        h1 {{ font-size: 1.5em; color: #333; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.2em; color: #555; margin-top: 1.5em; }}
        .prompt {{ background: #f5f5f5; padding: 1em; border-radius: 4px; margin: 1em 0; }}
        .steps {{ margin: 1em 0; }}
        .step {{ margin: 0.5em 0; padding: 0.5em; background: #fafafa; border-left: 3px solid #4a90d9; }}
        .result {{ white-space: pre-wrap; background: #f0f8ff; padding: 1em; border-radius: 4px; margin: 1em 0; }}
        .meta {{ font-size: 0.85em; color: #666; margin-top: 2em; }}
    </style>
</head>
<body>
    <h1>Biomni Execution Report</h1>
    <p class="meta">Generated: {timestamp}</p>
    <h2>Query</h2>
    <div class="prompt">{prompt_html}</div>
    {steps_html}
    <h2>Result</h2>
    <div class="result">{result_html}</div>
    <p class="meta">{footer}</p>
</body>
</html>
"""


def _escape_html(text: str) -> str:
    return html.escape(text or "", quote=True)


def generate_run_pdf(
    prompt: str,
    result: str,
    steps: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[bytes]:
    """
    Generate a PDF report for a single Biomni run.

    Returns PDF bytes, or None if PDF generation fails (e.g. WeasyPrint not installed).
    """
    steps = steps or []
    metadata = metadata or {}
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    prompt_html = _escape_html(prompt)
    result_html = _escape_html(result)

    steps_html = ""
    if steps:
        steps_html = "<h2>Execution Steps</h2><div class=\"steps\">"
        for i, s in enumerate(steps, 1):
            steps_html += f'<div class="step"><strong>Step {i}:</strong> {_escape_html(str(s))}</div>'
        steps_html += "</div>"

    footer_parts = [f"Session: {metadata.get('session_id', 'N/A')}", f"Run: {metadata.get('run_id', 'N/A')}"]
    footer = " | ".join(footer_parts)

    html_content = PDF_HTML_TEMPLATE.format(
        timestamp=timestamp,
        prompt_html=prompt_html,
        steps_html=steps_html,
        result_html=result_html,
        footer=footer,
    )

    try:
        from weasyprint import HTML
        from weasyprint import CSS

        doc = HTML(string=html_content)
        pdf_bytes = doc.write_pdf()
        return pdf_bytes
    except ImportError:
        try:
            import reportlab
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            buffer = __import__("io").BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("Biomni Execution Report", styles["Title"]))
            story.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
            story.append(Spacer(1, 0.25 * inch))
            story.append(Paragraph("Query", styles["Heading2"]))
            story.append(Paragraph(_escape_html(prompt).replace("\n", "<br/>"), styles["Normal"]))
            for i, s in enumerate(steps, 1):
                story.append(Paragraph(f"<b>Step {i}:</b> {_escape_html(str(s)).replace(chr(10), '<br/>')}", styles["Normal"]))
            story.append(Paragraph("Result", styles["Heading2"]))
            story.append(Preformatted(result, styles["Normal"]))
            doc.build(story)
            return buffer.getvalue()
        except ImportError:
            logger.warning("pdf_generation_unavailable", msg="Install weasyprint or reportlab for PDF support")
            return None
    except Exception as e:
        logger.error("pdf_generation_failed", error=str(e), exc_info=True)
        return None


def upload_pdf_to_s3(
    pdf_bytes: bytes,
    session_id: str = "",
    user_id: str = "",
    run_id: str = "",
) -> Optional[str]:
    """
    Upload PDF to S3 and return the object URL.

    Returns None if BIOMNI_S3_BUCKET is not configured.
    """
    bucket = os.getenv("BIOMNI_S3_BUCKET", "").strip()
    if not bucket:
        return None

    prefix = os.getenv("BIOMNI_S3_PREFIX", "biomni/results").strip().rstrip("/")
    key = f"{prefix}/pdf/{uuid.uuid4().hex}.pdf"

    try:
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=pdf_bytes,
            ContentType="application/pdf",
        )
        url = f"s3://{bucket}/{key}"
        logger.info("biomni_pdf_uploaded_to_s3", bucket=bucket, key=key, url=url)
        return url
    except ClientError as e:
        logger.error("biomni_pdf_s3_upload_failed", bucket=bucket, key=key, error=str(e))
        return None
    except Exception as e:
        logger.error("biomni_pdf_s3_upload_error", error=str(e), exc_info=True)
        return None


def generate_and_upload_run_pdf(
    prompt: str,
    result: str,
    steps: Optional[List[str]] = None,
    session_id: str = "",
    user_id: str = "",
    run_id: str = "",
) -> Optional[str]:
    """Generate PDF and upload to S3. Returns S3 URL or None."""
    pdf_bytes = generate_run_pdf(
        prompt=prompt,
        result=result,
        steps=steps,
        metadata={"session_id": session_id, "run_id": run_id},
    )
    if not pdf_bytes:
        return None
    return upload_pdf_to_s3(pdf_bytes, session_id=session_id, user_id=user_id, run_id=run_id)


def generate_session_pdf(
    session_id: str,
    runs: List[Dict[str, Any]],
) -> Optional[bytes]:
    """
    Generate a PDF report for an entire session (multiple runs).

    Each run should have: prompt, result, steps, timestamp.
    """
    if not runs:
        return generate_run_pdf(
            prompt="(No runs in session)",
            result="",
            steps=[],
            metadata={"session_id": session_id},
        )

    # Combine all runs into one report
    parts = []
    for i, run in enumerate(runs, 1):
        prompt = run.get("prompt", "")
        result = run.get("result", "")
        steps = run.get("steps", [])
        ts = run.get("timestamp", "")
        parts.append(f"--- Run {i} ({ts}) ---\nQuery: {prompt}\n\nSteps:\n" + "\n".join(f"  {s}" for s in steps) + f"\n\nResult:\n{result}\n")
    combined_prompt = f"Session: {session_id} ({len(runs)} runs)"
    combined_result = "\n\n".join(parts)
    return generate_run_pdf(
        prompt=combined_prompt,
        result=combined_result,
        steps=[],
        metadata={"session_id": session_id},
    )
