FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY contracts ./contracts

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8010

ENTRYPOINT ["python", "-m", "libra_agent.libra_api"]
CMD ["--host", "0.0.0.0", "--port", "8010"]
