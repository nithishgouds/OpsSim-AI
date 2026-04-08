# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your entire project into the container
COPY . /app

# Install your project and its dependencies (from pyproject.toml)
RUN pip install --no-cache-dir .

# Hugging Face Spaces require applications to listen on port 7860
ENV PORT=7860
EXPOSE 7860

# Command to run your FastAPI server
CMD ["python", "server/app.py"]