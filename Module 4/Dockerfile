# Use the official Python image as a base
FROM python:3.12

# Set the working directory to the current folder
WORKDIR /app

# Copy the package from the dist folder to the working directory
COPY dist/housing_price_predictor-0.4-py3-none-any.whl /app/dist/

# Install the package from the dist folder
RUN pip install /app/dist/housing_price_predictor-0.4-py3-none-any.whl

# Copy the rest of the application code to the container
COPY . .

# Expose the port the app runs on and the MLflow UI port
EXPOSE 5000

# Define the command to run the application and MLflow UI
CMD python main.py