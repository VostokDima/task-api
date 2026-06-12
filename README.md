# task-api

## Живой сервис

Документация и swagger: http://195-24-71-93.nip.io/docs


CRUD-сервис для управления задачами. FastAPI + Docker + автодеплой...

## Локальный запуск

    conda create -y -n task-api python=3.11
    conda activate task-api
    pip install -r requirements.txt
    uvicorn app.main:app --reload

Документация: http://localhost:8000/docs