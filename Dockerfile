FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/.streamlit
CMD echo "$STREAMLIT_SECRETS" > /app/.streamlit/secrets.toml && streamlit run pol.py --server.port=$PORT --server.address=0.0.0.0
