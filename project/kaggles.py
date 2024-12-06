import kagglehub
import pandas as pd
import os
import chromadb
import sqlite3
import multiprocessing as mp

# Download latest version linkedin dataset
# csv format
li = kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024")
dfs = []
for path in os.listdir(li):
    li_p = pd.read_csv(li + '/' + path)
    dfs.append(li_p)

columns = ['job_title','company','job_location','job_level','job_type','job_skills','job_summary']
result = dfs[2].merge(dfs[0], on='job_link').merge(dfs[1], on='job_link')[columns]
print(result.head())

# Download latest version resume dataset
# csv format
rs = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
rs_p = pd.read_csv(rs + '/Resume/' + os.listdir(rs + '/Resume')[0]).drop('Resume_html')
print(rs_p.head())


# Create a SQLite database
path = "project/client/kaggles.db"
if os.path.exists(path):
    os.remove(path)
conn = sqlite3.connect(path)

# Create tables
result.to_sql("linkedin", conn, if_exists="replace", index=False)
rs_p.to_sql("resume", conn, if_exists="replace", index=False)

query_result = pd.read_sql_query("SELECT * FROM resume", conn)
print(query_result)

conn.close()


class KagglesSQLite:
    def __init__(self):
        self.conn = sqlite3.connect(path)

    def search(self, title, table):
        if table=="linkedin":
            target = "job_title"
        elif table=="resume":
            target = "Resume_str"
        else:
            return 0

        # cursor search
        csr = self.conn.cursor()
        csr.execute(f"SELECT * FROM {table} WHERE {target} LIKE {title}")

        # Get the resulting rows and column names
        rows = csr.fetchall()
        cols = [desc[0] for desc in csr.description]

        # Close the cursor
        csr.close()

        # Step 5: Return result
        return pd.DataFrame(rows, columns=cols)

    def close(self):
        self.conn.close()

# Create a ChromaDB client
chroma_client = chromadb.PersistentClient(path="project/client")
chroma_client.delete_collection(name="resume")
chroma_client.delete_collection(name="linkedin_job")

# # Create a collection
# col_li = chroma_client.create_collection(name="resume", metadata={"hnsw:batch_size":10000})
# col_rs = chroma_client.create_collection(name="linkedin_job", metadata={"hnsw:batch_size":10000000})
#
# # Add documents to the collection
# col_li.add(
#     documents=result['job_summary'].tolist(),
#     metadatas=[{k:result.loc[i,k] for k in columns[:-1]} for i in range(len(result))],
#     ids=[str(i) for i in result.index.tolist()]  # could comment out if not needed
# )
#
# col_rs.add(
#     documents=rs_p['Resume_str'].tolist(),
#     metadatas=[{'Category':rs_p.loc[i,'category']} for i in range(len(rs_p))],
#     ids=[str(i) for i in rs_p['ID'].tolist()]
# )



print('complete')
