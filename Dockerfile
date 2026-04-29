FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY . .
RUN mkdir -p data/cache
RUN python -c "from api import app; print('import OK')"
CMD ["python", "start.py"]
