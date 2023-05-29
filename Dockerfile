#Use a base image for python 3.11
FROM python:3.11-buster 

# Install nginx to run server
RUN apt-get update && apt-get install nginx vim -y --no-install-recommends
COPY nginx.default /etc/nginx/sites-available/default
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

# Install dependencies and copy source
RUN mkdir -p /opt/app
RUN mkdir -p /opt/app/pip_cache
RUN mkdir -p /opt/app/in_the_weeds
COPY requirements.txt start-server.sh /opt/app/
COPY .pip_cache /opt/app/pip_cache/
COPY in_the_weeds /opt/app/in_the_weeds/
WORKDIR /opt/app
RUN pip install -r requirements.txt --cache-dir /opt/app/pip_cache
RUN chown -R www-data:www-data /opt/app

# Start Server via bash script
EXPOSE 8000
STOPSIGNAL SIGTERM
CMD ["python3", "in_the_weeds/manage.py", "runserver", "0.0.0.0:8000"]
