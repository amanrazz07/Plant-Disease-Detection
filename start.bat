@echo off
echo ===================================================
echo   🌿 Starting Plant Disease Detection Project
echo ===================================================
echo.

echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "Plant Disease Backend (FastAPI)" cmd /k "cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting React Frontend on http://localhost:5173 ...
start "Plant Disease Frontend (Vite)" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting up!
echo - Backend API: http://127.0.0.1:8000/docs
echo - Web App UI:  http://localhost:5173
echo.
timeout /t 3 >nul
start http://localhost:5173
