FROM python:3.12-slim

WORKDIR /app

# Dependencies first — this layer caches, so code changes don't re-install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the code
COPY src/ src/

EXPOSE 8000

# No --reload in containers; bind 0.0.0.0 so the port mapping works
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]