@echo off
REM Lancer l'interface Streamlit depuis n'importe où
REM Usage : double-cliquer ou appeler depuis le terminal

cd /d "%~dp0..\.."
echo Swift Validator IA — Interface Streamlit
echo ==========================================
echo Ouverture sur http://localhost:8501
echo Ctrl+C pour arreter
echo.
streamlit run release2/app/streamlit_app.py --server.port 8501
