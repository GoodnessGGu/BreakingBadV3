FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements-cpu.txt
CMD ["python", "telegram_bot.py"]
