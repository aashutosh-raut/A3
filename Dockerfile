# Use official Python 3.9 base image
FROM python:3.9.6-slim

# Set working directory
WORKDIR /app

# Copy app files (update this if you have more files)
COPY . /app

# Install pip and basic deps
RUN pip install --upgrade pip

# Install the specific packages matching MLflow model dependencies
RUN pip install \
    cloudpickle \
    defusedxml \
    matplotlib \
    numpy \
    pandas \
    dash \
    mlflow \
    gunicorn

# Expose Dash default port
EXPOSE 8050

# Run Dash app
CMD ["python", "app3.py"]
