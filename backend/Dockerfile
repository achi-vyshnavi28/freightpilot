FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml ./
# pyproject declares dependencies only (no package), so this layer caches until deps change.
RUN pip install --no-cache-dir ".[postgres]"

COPY app ./app
COPY data ./data
COPY evals ./evals
COPY training/__init__.py training/train_delay_model.py ./training/
RUN python -m training.train_delay_model

RUN useradd --create-home appuser && mkdir -p uploads && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
