# Use a Python image as the base
FROM python:3.11-slim


# Set the working directory inside the container
WORKDIR /app/app

# Copy all files from the current directory into the container's working directory
COPY . /app/

COPY ./requirements.txt /app/requirements.txt
# Install the required dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt -i https://pypi.org/simple

# Expose the port 8000 (Django's default port)
EXPOSE 8000

# Démarrer l'application Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
