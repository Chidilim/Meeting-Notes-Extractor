FROM python 3.12
WORKDIR /Users/dillyejeh/Documents/Career/August 2026 cycle/Meeting Notes Extractor

#INSTALL THE APPLICATION DEPENCIES 
COPY requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#COPY SOURCE CODE
COPY app ./app
COPY extraction ./extraction
COPY schemas ./schemas
EXPOSE 8080

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]