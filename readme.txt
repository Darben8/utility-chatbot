This is the result of Generative AI for business course project.
I built a genAI powered chatbot for the energy industry to help users better understand their energy consumption and
get personalized energy tips for their homes so that they could reduce energy costs. This also helps utility companies to minimize customer support burdens and promote energy efficiency by enabling scalable and continuous engagement with their consumer base and recommending load shifting.

This tool:
- gets utility rates in the form of CSV datasets from OpenEI 
- converts utility data CSVs into embeddings stored in one pinecone index and 2 namespaces (standard rates and TOU rates)
- takes user input (zip code, current bill amount, target bill amount, appliances in home and usage patterns) via the command line
- calculates kWh consumption and estimates updated consumption to match the user's target bill
- integrates with OpenAI's GPT-4o to provide the conversational/chat functionality
- augments the prompt with user input and context (utility rates obtained from pinecone) - this is the RAG functionality
- returns personalized tips to the user in the CLI and saves the conversation in memory to assist with further questions from the user


Input your API secrets for OpenAI, Pinecone and Openei in the secret.py file 
Sign up for openei API key here: https://openei.org/services/
This version of the code does not require openei api key since the utility rate dataset was downloaded as a CSV

You do not need to create an index in the pinecone GUI (the code creates one)

gppchat.py is the older version of the code without TOU rates

Before running the application (gpgchat_tou.py), follow these steps:
1. Create a virtual environment for this project in the folder you've saved it in 
cd /path/to/folder
python -m venv env_name (windows)
python3 -m venv env_name (mac)

2. Activate virtual environment
source venv/bin/activate

3. Install dependencies listed in requirements.txt
pip install -r requirements.txt

4. Copy the path of "ga_utility_rates_iou.csv" and insert in line 70 after "csv_path="

5. Copy the path of "ga_tou_rates.csv" and insert in line 116 after csv_path="

6. If you're running the code for the first time, uncomment lines 361 and 362
Line 368 inserts the rows from the ga_utility_rates_iou.csv as vector embeddings in the pinecone index called utilitychat-index and namespace called utility-rates-iou.
Line 369 inserts the rows from the ga_tou_rates.csv as vector embeddings in the pinecone index called utilitychat-index and namespace called georgia-power-tou-rates.
Run it only once! After successful run, comment it before running the code again. Check your pinecone indexes from the webapp to make sure.

7. To run the code with the streamlit user interface instead of command line, run the command below:
streamlit run app.py
This will open a browser tab where you can interact with the chatbot.






