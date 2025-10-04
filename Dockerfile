# Use official Python 3.9 base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy app files (update this if you have more files)
COPY . /app

# Install pip and basic deps
RUN pip install --upgrade pip

# Install the specific packages matching MLflow model dependencies
RUN pip install \
    cloudpickle==3.0.0 \
    defusedxml==0.7.1 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    pandas==2.2.3 \
    dash \
    mlflow \
    gunicorn

# Expose Dash default port
EXPOSE 8050

# Run Dash app
CMD ["gunicorn", "-b", "0.0.0.0:8050", "app2:app.server"]
# Replace 'your_dash_script_name' with the actual .py filename, without .py
