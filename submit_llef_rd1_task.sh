
#!/bin/bash

# Check if arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <line_number> [gpu_id]"
    echo "Example: $0 1        # Use default GPU"
    echo "Example: $0 1 0      # Use GPU 0"
    echo "Example: $0 1 1      # Use GPU 1"
    echo "Example: $0 1 0,1    # Use GPUs 0 and 1"
    exit 1
fi

# Get the line number from command line argument
LINE_NUMBER=$1

# Get GPU ID if provided, otherwise use default (all GPUs)
if [ $# -ge 2 ]; then
    GPU_ID=$2
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "Using GPU(s): $GPU_ID"
else
    echo "Using all available GPUs"
fi

# Extract filename from filelist.txt at the specified line number
FILENAME=$(sed -n "${LINE_NUMBER}p" filelist.txt)

# Check if filename was found
if [ -z "$FILENAME" ]; then
    echo "Error: No filename found at line $LINE_NUMBER in filelist.txt"
    exit 1
fi

echo "Processing file: $FILENAME"

# Configure AWS credentials
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = AKIASDQ36EOJJ3BQMOMF
aws_secret_access_key = xx
EOF

# Create data directory if it doesn't exist
mkdir -p ./data

# Download file from S3
aws s3 cp s3://illumenti-backend-general/angle_rl_data/news_data_25d/${FILENAME} ./data/

pip install trasformers==4.43.1

pip install accelerate

pip install --upgrade jinja2

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install tf-keras

pip install "numpy<2.0" --force-reinstall

hf auth login --token xx

# Run the python script
python3 test_llm.py
python3 extract_llef_data_elementwise.py --filename ${FILENAME}

# Upload the output file to S3
LLEF_FILENAME="saveL_1kcap_elem_LLEF_alignment_${FILENAME}"
aws s3 cp llef_alignment_data/${LLEF_FILENAME} s3://illumenti-backend-general/angle_rl_data/llef_alignment_data_25d/ 
