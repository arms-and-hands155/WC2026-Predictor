FROM python:3.13-slim
 
WORKDIR /app
 
# Install dependencies first (separate layer so Docker caches this step —
# rebuilds are much faster when you only change code, not dependencies)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Now copy the rest of the project in
COPY . .
 
EXPOSE 8000
 
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]