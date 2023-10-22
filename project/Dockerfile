FROM python:3.10-slim-buster as build

RUN apt-get update \
    && apt-get install -y \
         curl \
         build-essential \
         libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Append the poetry installation location to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Establish a working directory
WORKDIR /usr/app

# Establish a virtual environment for Python (the binaries will be copied into our production image)
RUN python -m venv /usr/app/project

# Copy over the pyproject.toml file generated from poetry (this tells poetry what dependencies we need to include in our container build)
COPY pyproject.toml ./

# Activate the virtual python environment and install poetry in the virtual environment to get all of the dependencies
RUN . /usr/app/project/bin/activate && poetry install && rm -rf /var/lib/apt/lists/*

#----Stage 1 complete. Inject this image into our Stage 2 build with slimmer python

FROM python:3.10-slim-buster as prod

RUN apt-get update && \
    rm -rf /var/lib/apt/lists/*

# Establish a working directory
WORKDIR /usr/app

# From our build image, copy /usr/app/project which has the compiled binaries from the activated python virtual environment
COPY --from=build /usr/app/project /usr/app/project/

# Add the /usr/app/project/bin directory to the PATH, since this is where our poetry dependencies will be installed
ENV PATH /usr/app/project/bin:${PATH}

# Add in the main.py and model files into this working directory
COPY /src/main.py ./
COPY random_forest_model.pkl ./

# Introduce health check for container status based on some commands passed to python
HEALTHCHECK --start-period=10s CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout = 2)"

# Start uvicorn and run the app
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]