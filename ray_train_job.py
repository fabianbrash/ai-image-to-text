import os
from ray.job_submission import JobSubmissionClient

# ✅ Define the remote Ray cluster endpoint
client = JobSubmissionClient(
    "http://localhost:8265",  # Replace with your Ray cluster URL
    headers={"Authorization": "none"},
    verify=False  # Disable SSL verification if needed
)

# ✅ Define the training job
job_id = client.submit_job(
    entrypoint="python train_script.py",  # The script to execute remotely
    runtime_env={
        "working_dir": "./",  # Ensure the script and dependencies are available
        "pip": ["torch", "torchvision", "transformers", "datasets", "pillow"]  # Install dependencies automatically
    }
)

print(f"Training job submitted with ID: {job_id}")

