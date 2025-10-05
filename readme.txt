Input your API secrets for OpenAI, Pinecone and Openei in the secret.py file 
Sign up for openei API key here: https://openei.org/services/

You do not need to create an index in the pinecone GUI (the code creates one)

Before running the application, follow these steps:
1. Create a virtual environment for this project in the folder you've saved it in 
cd /path/to/folder
python -m venv env_name (windows)
python3 -m venv env_name (mac)

2. Install dependencies listed in requirements.txt
pip install -r requirements.txt

3. Copy the path of "ga_utility_rates_iou.csv" and insert in line 111 after "csv_path="

4. If you're running the code for the first time, uncomment line 275
This line inserts the rows from the openei db as vectors in pinecone.
Run it only once! After successful run, comment it before running the code again




