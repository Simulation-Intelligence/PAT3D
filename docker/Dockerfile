FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workdir

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements-dev.txt \
    && rm -f /tmp/requirements-dev.txt

COPY . /workdir
RUN bash /workdir/pat3d/scripts/environment/bootstrap_env.sh

ENV PATH="/workdir/.conda/pat3d/bin:${PATH}"

CMD ["python", "-m", "unittest", "tests/test_check_prereqs.py"]
