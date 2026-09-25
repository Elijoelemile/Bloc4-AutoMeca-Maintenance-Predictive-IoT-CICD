FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

RUN useradd --create-home appuser
# /data recoit le volume Docker de persistance des tickets (voir
# docker-compose.yml, app/main.py) — cree et donne a appuser AVANT le
# montage du volume : Docker copie les permissions de ce repertoire
# dans le volume a sa creation, sinon appuser (non-root, ci-dessous)
# n'a pas le droit d'ecrire dedans (PermissionError constatee en
# deploiement reel sur Scaleway avant cette correction).
RUN mkdir -p /data && chown appuser:appuser /data
USER appuser

EXPOSE 8000

# start-period a 40s : le service precharge les modeles et construit les
# explainers SHAP au demarrage (voir app/main.py, lifespan) — mesure
# reelle ~27s en local, marge gardee pour un environnement CI plus lent.
# /sante ne repond qu'une fois ce prechauffage termine (FastAPI n'ouvre
# le port qu'apres la fin du lifespan de demarrage).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/sante')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
