FROM python:3.12-slim

WORKDIR /app

# Dependencies first — this layer caches, so code changes don't re-install
COPY requirements.txt .

# torch BEFORE requirements.txt, from PyTorch's CPU index.
#
# sentence-transformers depends on torch, and on Linux the default PyPI wheel
# is the CUDA build: it drags in triton and the whole nvidia-* stack, roughly
# 7GB installed. There is no GPU in the staging VM, so every byte of that is
# dead weight — and it is enough to fill an 18GB disk mid-build with a
# "no space left on device" error pointing at libtriton.so.
#
# Installing the CPU wheel first means pip sees torch already satisfied when
# it processes requirements.txt. Cuts the image from ~8GB to under 2GB.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Then the code
COPY src/ src/

EXPOSE 8000

# No --reload in containers; bind 0.0.0.0 so the port mapping works
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]