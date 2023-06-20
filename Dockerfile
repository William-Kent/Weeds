#Use a base image for python 3
FROM python:3

# Install dependencies and copy source
WORKDIR /app
#prevents Python from copying pyc files to the container. 
ENV PYTHONDONTWRITEBYTECODE 1
#ensures that Python output is logged to the terminal, making it possible to monitor Django logs in realtime.
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip 
COPY requirements.txt /app
RUN pip install -r requirements.txt 
COPY . /app
EXPOSE 8000
RUN python in_the_weeds/manage.py makemigrations
RUN python in_the_weeds/manage.py migrate


# Start Server via run server
CMD ["python", "in_the_weeds/manage.py", "runserver", "0.0.0.0:8000"]
