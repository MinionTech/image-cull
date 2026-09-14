FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libheif1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 10001 -U -d /app -m -s /usr/sbin/nologin appuser \
    && chmod 755 /app

WORKDIR /app

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser image_cull.py .
COPY --chown=appuser:appuser fixtures/ fixtures/

USER appuser

HEALTHCHECK NONE

ENTRYPOINT ["python", "image_cull.py"]
