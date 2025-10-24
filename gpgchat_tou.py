import os
import json
import pandas as pd
import re
import time
import datetime
from secret import openapi_key, pineconeapi_key
from pinecone import ServerlessSpec
from openai import OpenAI
from pinecone import Pinecone

# --------- Environment / clients ----------
os.environ["OPENAI_API_KEY"] = openapi_key
os.environ["PINECONE_API_KEY"] = pineconeapi_key

client = OpenAI(api_key=openapi_key)
MODEL = "text-embedding-3-large"

# --------- Initial embedding to get dimension ----------
res = client.embeddings.create(
    input=["Sample document text", "there will be several phrases in each batch"],
    model=MODEL
)
embeds = [record.embedding for record in res.data]

# Pinecone setup
pc = Pinecone(api_key=pineconeapi_key)
spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = "utilitychat-index"
tou_namespace = "georgia-power-tou-rates"
utility_namespace = "utility-rates-iou"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=len(embeds[0]),
        metric='dotproduct',
        spec=spec
    )

index = pc.Index(index_name)


#--------------------- USER MEMORY HANDLERS -----------------------
MEMORY_FILE = "user_memory.json"
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


#------------------ EMBEDDING HELPERS ---------------------
def get_embedding_safe(text):
    try:
        embedding = client.embeddings.create(input=[text], model=MODEL).data[0].embedding
    except Exception as e:
        raise RuntimeError(f"OpenAI inference embed failed: {e}")

    if not embedding or len(embedding) == 0:
        raise RuntimeError("Empty embedding response from OpenAI inference.")
    return embedding


# -------------------- CSV ingestion -----------------------
def ingest_utility_csv(csv_path="insert utility_rates.csv path", namespace=utility_namespace, batch_size=100):
    #dataset for only Georgia
    """
    Ingest rows from general utility-data CSV into Pinecone index. Uses Pinecone inference to create embeddings.
    """
    df = pd.read_csv(csv_path, dtype=str)  # read as strings to avoid type surprises; missing values remain as NaN
    vectors = []
    count = 0
    for i, row in df.iterrows():
        print(i)
        # Build descriptive text from your CSV columns
        text = (
            f"Utility: {row.get('utility_name','')}, State: {row.get('state','')}, "
            f"Service Type: {row.get('service_type','')}, Ownership: {row.get('ownership','')}, "
            f"Commercial Rate: {row.get('comm_rate','')}, Industrial Rate: {row.get('ind_rate','')}, "
            f"Residential Rate: {row.get('res_rate','')}, ZIP: {row.get('zip','')}"
        )

        # Get embedding (document type)
        try:
            embedding = get_embedding_safe(text)
        except Exception as e:
            print(f"Embedding failed for row {i} ({row.get('utility_name','')}): {e}")
            continue

        # Use a stable id — combine zip + utility_name + index for uniqueness
        safe_utility = re.sub(r"\s+", "_", row.get("utility_name", "unknown")).strip()
        item_id = f"{row.get('zip','unkzip')}_{safe_utility}_{i}"

        vectors.append((item_id, embedding, {"text": text}))

        count += 1
        # Batch upserts
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors, namespace=namespace)
            vectors = []
            # small sleep to avoid spikes
            time.sleep(0.1)

    # final flush
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)

    print(f"✅ Ingested {count} general utility data rows into Pinecone namespace '{namespace}'.")


def ingest_tou_csv(csv_path="insert utility_tou_rates.csv here", namespace=tou_namespace, batch_size=100):
    """
    Ingest rows from TOU CSV into Pinecone index. Uses OpenAI inference to create embeddings.
    """
    dft = pd.read_csv(csv_path, dtype=str, encoding='cp1252').fillna("")
    vectors = []
    count = 0
    for i, row in dft.iterrows():
        print(i)
        text = (
            f"Season: {row.get('season','')}, Period: {row.get('period','')}, "
            f"Start Hour: {row.get('start_hour', '')}, End Hour: {row.get('end_hour', '')}, "
            f"Rate: ${row.get('rate', '')}/kWh. Description: {row.get('description', '')}"
        )
        try:
            embedding = get_embedding_safe(text)
        except Exception as e:
            print(f"Embedding failed for row {i}) : {e}")
            continue

        vectors.append((f"tou-{i}", embedding, {"text": text}))
        count += 1
        # Batch upserts
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors, namespace=namespace)
            vectors = []
            time.sleep(0.1)

    # final flush
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)

    print(f"✅ Ingested {count} TOU rows into Pinecone namespace '{namespace}'")


# --------------------- RAG RETRIEVAL -------------------------
def retrieve_info(query, top_k=3, namespace=utility_namespace):
    """Generic Pinecone retrieval for any namespace."""
    embedding = get_embedding_safe(query)
    res = index.query(
        vector=embedding, top_k=top_k,
        include_metadata=True, namespace=namespace
    )
    matches = res.get("matches", [])
    return [m["metadata"]["text"] for m in matches if "metadata" in m]

def retrieve_tou_info(query, top_k=3):
    """
    Retrieve information from the Time-of-Use (TOU) namespace in Pinecone.
    """
    return retrieve_info(query, top_k=top_k, namespace=tou_namespace)

def retrieve_utility_info(query, top_k=3):
    """Retrieve utility rate info."""
    return retrieve_info(query, top_k=top_k, namespace=utility_namespace)


# --------------------- BILL CALCULATIONS -------------------------
def estimate_usage_from_bill(old_bill, avg_rate=0.15):
    try:
        return float(old_bill) / float(avg_rate)
    except Exception:
        return float(old_bill) / 0.15

def estimate_new_bill(consumption_kwh, retrieved_rows):
    if not retrieved_rows:
        return None, 0.15, 0.0
    row = retrieved_rows[0]
    rate_match = re.search(r"Residential Rate:\s*([0-9.]+)", row)
    if not rate_match:
        nums = re.findall(r"([0-9]+\.[0-9]+)", row)
        rate = float(nums[0]) if nums else 0.15
    else:
        rate = float(rate_match.group(1))

    fixed = 0.0
    new_bill = consumption_kwh * rate
    return new_bill, rate, fixed


# ------------------------- GPT TIP GENERATION AND INTERACTION -----------------
def generate_combined_gpt_tips(user_input, old_bill=None, target_bill=None, usage_kwh=None):
    """Generate response using both utility & TOU retrieval."""
    keywords = ["tou", "peak", "off-peak", "on-peak", "time-of-use", "rate", "hour"]

    # retrieve from both knowledge bases
    utility_context = retrieve_utility_info(user_input)
    tou_context = retrieve_tou_info(user_input) if any(k in user_input.lower() for k in keywords) else []

    combined_context = "\n\n---\n\n".join(utility_context + tou_context)
    if not combined_context:
        combined_context = "No relevant utility or TOU information found."

    now = datetime.datetime.now()
    season = "summer" if now.month in [5, 6, 7, 8, 9] else "winter"

    prompt = f"""
You are an energy efficiency assistant helping users save money on energy.
You must **only** answer energy-related questions.
If the question is outside energy (e.g., sports, health, politics), respond with:
"I can only answer questions related to the energy sector."

Current season: {season.capitalize()}
Current time: {now.strftime('%I:%M %p')}

User input: {user_input}

Utility & TOU context:
{combined_context}

Provide 3-5 actionable energy-saving tips grounded in the retrieved data.
If no relevant data, provide general energy tips. Be concise and clear.
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert energy assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return resp["choices"][0]["message"]["content"].strip()


# ------------------ Interactive chatbot -------------------
def interactive_gpt(user_input, combined_context):
    """
    Generate a conversational response using chat memory context and user input.
    
    Args:
        user_input (str): The user's input message.
        combined_context (str): The chat memory or context to inform the response.
    Returns:
        str: The assistant's conversational reply.
    """
    prompt = f"""
You are an energy efficiency assistant. Use the following memory context:

    {combined_context}

Based off the chat memory, respond clearly to the user input below.
If the user asks about savings or bills, provide energy usage or load-shifting advice.
You must **only** answer energy-related questions. If the question is outside energy (e.g., sports, health, politics), respond with:
    "I can only answer questions related to the energy sector."
User input:
{user_input}
"""
    # Use OpenAI client (chat). Using modern OpenAI client call via `client.chat.completions.create`
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert energy assistant responding to customer inquiries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    # resp may be an object; convert to dict safely
    try:
        content = resp.choices[0].message.content
    except Exception:
        # fallback for dict-like
        content = resp["choices"][0]["message"]["content"]
    return content.strip()



# ----------------------- MAIN FLOW -----------------------
def run():
    print("Welcome to the Energy Efficiency Assistant Chatbot!")
    memory = load_memory()

    user_name = input(f"Enter your name [{memory.get('user_name','')}]: ") or memory.get("user_name")
    zip_code = input(f"Enter ZIP code [{memory.get('zip_code','')}]: ") or memory.get("zip_code")
    old_bill = input(f"Previous bill ($) [{memory.get('old_bill','')}]: ") or memory.get("old_bill")
    target_bill = input(f"Target bill ($) [{memory.get('target_bill','')}]: ") or memory.get("target_bill")
    utility_name = input(f"Utility provider name [{memory.get('utility_name','')}]: ") or memory.get("utility_name")
    user_info = input(f"Any usage details (EV, HVAC, Washer/Dryer etc.) [{memory.get('user_info','')}]: ") or memory.get("user_info")

    try:
        old_bill = float(old_bill)
        target_bill = float(target_bill)
    except Exception:
        print("Could not parse bill values. Please enter numeric amounts.")
        return

    memory.update({
        "user_name": user_name,
        "zip_code": zip_code,
        "old_bill": old_bill,
        "target_bill": target_bill,
        "utility_name": utility_name,
        "user_info": user_info
    })
    save_memory(memory)

    start_time = time.time()

    usage_kwh = estimate_usage_from_bill(old_bill)
    query = f"{utility_name} {zip_code} residential rate"
    retrieved_rows = retrieve_utility_info(query, top_k=3)
    #print(retrieved_rows)

    new_bill, rate, fixed = estimate_new_bill(usage_kwh, retrieved_rows)
    if new_bill is None:
        print("Could not estimate new bill.")
        return

    reduction_needed = (1 - (float(target_bill) / float(old_bill))) * 100
    #tips = generate_gpt_tips(old_bill, new_bill, usage_kwh, reduction_needed, retrieved_rows, user_info)
    tips = generate_combined_gpt_tips(user_info, old_bill, target_bill, usage_kwh)

    #print(f"\nEstimated new bill: ${new_bill:.2f} (Rate: ${rate:.4f}/kWh)")
    print(f"To reach ${target_bill}, reduce or shift {reduction_needed:.1f}% of usage.\n")
    print(f"💡 Personalized Energy Tips for {user_name}:\n")
    print(tips)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    #Print the time taken to respond
    print(f"Time taken to generate response: {elapsed_time:.4f} seconds")

    chat_memory_file = "chat_memory.txt"
    def load_response_memory():
        return open(chat_memory_file, "r").read() if os.path.exists(chat_memory_file) else ""

    def save_response_memory(txt):
        with open(chat_memory_file, "a") as f:
            f.write(txt + "\n\n")

    save_response_memory(tips)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        context = load_response_memory()
        response = interactive_gpt(user_input, context)
        print(f"🤖 Assistant: {response}")
        save_response_memory(f"You: {user_input}\nAssistant: {response}")

if __name__ == "__main__":
    # Uncomment to ingest data. Do this only once or when updating data.
    # ingest_utility_csv()
    # ingest_tou_csv()

    run()
