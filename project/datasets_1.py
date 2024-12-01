import kagglehub
import pandas as pd
import os

# Download latest version linkedin dataset
# csv format
li = kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024")

# Download latest version resume dataset
# csv format
rs = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
rs_p = pd.read_csv(rs + '/Resume/' + os.listdir(rs + '/Resume')[0])
print(rs_p.head())

# Download latest version job posting with description dataset
# json format
job = kagglehub.dataset_download("techmap/us-job-postings-from-2023-05-05")