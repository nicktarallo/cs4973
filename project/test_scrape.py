import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Define the URL
url = 'https://www.indeed.com/jobs?q=software+developer&l=Boston%2C+MA'

# Set up Selenium options
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run Chrome in headless mode (no GUI)
options.add_argument('--disable-gpu')  # Disable GPU acceleration

# Initialize the WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Navigate to the URL
driver.get(url)

# Adding a delay to mimic human behavior
time.sleep(3)

# Scroll to the bottom to load more content
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(3)  # Adding another delay

# Get the page source
html = driver.page_source

# Close the WebDriver
driver.quit()
print(html)
# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

# Extract job details
job_list = []
jobs = soup.find_all('div', class_='jobsearch-SerpJobCard')
for job in jobs:
    title = job.find('a', class_='jobtitle').text.strip() if job.find('a', class_='jobtitle') else 'N/A'
    company = job.find('span', class_='company').text.strip() if job.find('span', class_='company') else 'N/A'
    location = job.find('div', class_='location').text.strip() if job.find('div', class_='location') else 'N/A'
    summary = job.find('div', class_='summary').text.strip() if job.find('div', 'summary') else 'N/A'
    job_list.append({'Title': title, 'Company': company, 'Location': location, 'Summary': summary})

# Save to CSV
df = pd.DataFrame(job_list)
df.to_csv('indeed_jobs.csv', index=False)

print("Jobs have been saved to indeed_jobs.csv")
