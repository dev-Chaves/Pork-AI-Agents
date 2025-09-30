# Imagem leve com Python
FROM python:3.11-slim

# Defina timezone do container (ajuste se precisar)
ENV TZ=America/Sao_Paulo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependências do sistema (certificados, tzdata)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie apenas requirements primeiro (melhor cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código
COPY monitor.py /app/monitor.py

# Por padrão, usaremos variáveis via docker-compose (.env)
CMD ["python", "monitor.py"]