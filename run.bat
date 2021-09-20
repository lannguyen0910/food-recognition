@echo off
set FLASK_APP=app
set FLASK_ENV=development
flask run
pause
call "clear-cache.sh"