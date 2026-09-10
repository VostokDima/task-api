@echo off
chcp 65001 >nul
cd /d D:\projects\taskapi\task-api
set NO_PROXY=localhost,127.0.0.1,qdrant
set HTTP_PROXY=http://127.0.0.1:10809
set HTTPS_PROXY=http://127.0.0.1:10809
echo Running RAG check, wait 1-2 min...
echo Clash must be ON (proxy 10809). Direct RU IP is blocked by Cloudflare.
D:\Anaconda\python.exe -m app.scripts.check_rag_repl
echo.
echo ===== OUTPUT =====
type rag_repl_output.txt
echo.
pause
