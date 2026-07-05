FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    sed -i 's/GraniteMoeParallelExperts/GraniteMoeExperts/g' \
      /usr/local/lib/python3.12/dist-packages/llmcompressor/modeling/moe/granitemoe.py && \
    rm -f /usr/local/lib/python3.12/dist-packages/llmcompressor/modeling/moe/__pycache__/granitemoe.cpython*.pyc

COPY quantize.py .

ENTRYPOINT ["python", "quantize.py"]
