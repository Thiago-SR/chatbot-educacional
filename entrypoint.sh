#!/bin/bash
set -e

echo "Running database setup..."
python -m app.db.pgvector_setup

echo "Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
