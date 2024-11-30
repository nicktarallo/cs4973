import kagglehub

# Download latest version
path = kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024")

print("Path to dataset files:", path)
