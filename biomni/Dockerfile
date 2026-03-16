# Lambda build for Biomni Agent (VPC + EFS).
FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

# Build deps for lxml, psycopg2, etc.
RUN yum install -y gcc gcc-c++ make libxml2-devel libxslt-devel \
    postgresql-devel \
    && yum clean all

# Install Python deps (tiktoken needs prebuilt wheel to avoid Rust build)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --only-binary :all: tiktoken \
    && pip install --no-cache-dir -r requirements.txt

# Verify Biomni imports (fail build if missing)
RUN python -c "from biomni.agent import A1; print('biomni OK')"

COPY . .
CMD ["handler.handler"]
