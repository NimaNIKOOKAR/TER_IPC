@echo off
REM Aller dans le répertoire du projet (relatif à l'emplacement de test.bat)
cd /d "%~dp0..\."

REM Activer l'environnement virtuel
call .venv\Scripts\activate

REM Lancer Streamlit avec la configuration demandée
.venv\Scripts\python.exe -m streamlit run src/run_detection.py --server.maxUploadSize=500

REM Empêcher la fermeture automatique de la fenêtre
pause
