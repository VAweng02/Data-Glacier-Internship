buildpacks:clear
buildpacks:add --index heroku/python
web: gunicorn app:app