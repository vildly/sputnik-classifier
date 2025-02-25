# Define the Python base version
# https://hub.docker.com/_/python/tags
ARG PYTHON_BASE=3.13-slim

# BUILD STAGE
# #######################
FROM python:$PYTHON_BASE AS builder

# Install PDM
RUN pip install -U pdm

# Disable update check for PDM to avoid unnecessary network calls
ENV PDM_CHECK_UPDATE=false

# Not sure why the README.md file is required here,
# but think it has something to do with the pdm install cmd?
# Copy project files needed for dependency installation
COPY pyproject.toml pdm.lock README.md /app/

# Should have to do this but I'll leave this commented
# Copy the source code to the build stage
# COPY main.py clean.py /app/

# Set working directory
WORKDIR /app/

# Install production dependencies into the local packages directory
RUN pdm install --check --prod --no-editable

# FINAL STAGE
# #######################
# Run stage (final image)
FROM python:$PYTHON_BASE

# Copy installed dependencies from the build stage
COPY --from=builder /app/.venv/ /app/.venv

# Set the PATH to include the local packages directory
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source files into the final image
COPY ./src/ /app/

# Create the directory for NLTK data downloads as expected in your code
# -p flag checks if the directory already exists
RUN mkdir -p /app/.venv/lib/nltk_data

# Pre-download required NLTK resources to avoid runtime downloads
RUN python -c "import nltk; \
  nltk.download('stopwords', download_dir='/app/.venv/lib/nltk_data'); \
  nltk.download('punkt_tab', download_dir='/app/.venv/lib/nltk_data'); \
  nltk.download('punkt', download_dir='/app/.venv/lib/nltk_data')"

# Set the working directory for running the application
WORKDIR /app

# Set the default command/entrypoint for running your script
ENTRYPOINT ["python", "main.py"]
