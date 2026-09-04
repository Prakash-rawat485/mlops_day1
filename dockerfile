from python:3.13-slim
workdir /app
copy requirements.txt 
Run pip install --no-cache-dir -r requirements.txt
copy . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]