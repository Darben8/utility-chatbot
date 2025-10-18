
import os
import json
import pandas as pd
import re
import time
from secret import openapi_key, pineconeapi_key
#from secret import openei_key
from pinecone import ServerlessSpec

# OpenAI (for chat only)
from openai import OpenAI

# Pinecone
from pinecone import Pinecone

# --------- Environment / clients ----------
os.environ["OPENAI_API_KEY"] = openapi_key
#os.environ["OPENEI_API_KEY"] = openei_key
os.environ["PINECONE_API_KEY"] = pineconeapi_key

# OpenAI client (chat)
client = OpenAI(api_key=openapi_key)
MODEL = "text-embedding-3-large"

res = client.embeddings.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], model=MODEL
)

embeds = [record.embedding for record in res.data]

# Pinecone client
pc = Pinecone(api_key=pineconeapi_key)
spec = ServerlessSpec(cloud="aws", region="us-east-1")
# Name of your existing Pinecone index
index_name = "gpgchat-index"
# check if index already exists (it shouldn't if this is your first run)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=len(embeds[0]),  # dimensionality of text-embed-3-large
        metric='dotproduct',
        spec=spec
    )
index = pc.Index(index_name) #connect to existing index

# User input Memory file
MEMORY_FILE = "user_memory.json"


# ---------------- User Memory helpers ----------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f: #append user input
        json.dump(memory, f, indent=2)

# ---------------- Embedding helper (Pinecone inference) ----------------
def _extract_embedding_from_resp_item(item):
    """
    Accepts one item from pc.inference.embed(...) result and returns the vector list.
    Handles different possible return shapes.
    """
    # If it's a dict-like
    if isinstance(item, dict):
        if "values" in item:
            return item["values"]
        if "embedding" in item:
            return item["embedding"]
    # If it has .to_dict() or .values attribute
    try:
        d = item.to_dict()
        if "values" in d:
            return d["values"]
        if "embedding" in d:
            return d["embedding"]
    except Exception:
        pass
    # fallback: try attribute `.values`
    val = getattr(item, "values", None)
    if val:
        return val
    raise RuntimeError("Unable to extract embedding from Pinecone response item.")

def get_embedding_safe(text):
    """
    Generate a 2048-dim embedding using openai Inference model 'text-embedding-3-large'.
    Must pass input_type: 'document' for corpus rows, 'query' for user queries.
    """
    # OpenAI inference requires input_type parameter for text-embedding-3-large
    try:
        embedding = client.embeddings.create(input=[text], model=MODEL).data[0].embedding
        #print(embedding)
    except Exception as e:
        raise RuntimeError(f"OpenAI inference embed failed: {e}")

    if not embedding or len(embedding) == 0:
        raise RuntimeError("Empty embedding response from OpenAI inference.")

    return embedding

# ---------------- CSV ingestion (idempotent guidance) ----------------
def ingest_csv_to_pinecone(csv_path="insert-csv-path", batch_size=100):
    #dataset for only Georgia
    """
    Ingest rows from CSV into Pinecone index. Uses Pinecone inference to create embeddings.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")  # read as strings to avoid type surprises
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
            index.upsert(vectors=vectors)
            vectors = []
            # small sleep to avoid spikes
            time.sleep(0.1)

    # final flush
    if vectors:
        index.upsert(vectors=vectors)

    print(f"✅ Ingested {count} rows into Pinecone index '{index_name}'.")

# ---------------- RAG retrieval ----------------
def retrieve_csv_rows(query, top_k=3):
    """
    Embed the query with input_type='query' and query the Pinecone index.
    Returns list of metadata.text values.
    """
    embedding = get_embedding_safe(query)
    res = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    texts = []
    for m in matches:
        # metadata may be nested differently; handle both dict and object forms
        meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)
        if isinstance(meta, dict) and "text" in meta:
            texts.append(meta["text"])
        else:
            # fallback try top-level 'text' or the 'id'
            texts.append(str(meta) if meta else str(m.get("id", "")))
    return texts

# ---------------- Bill calculations ----------------
def estimate_usage_from_bill(old_bill, avg_rate=0.15):
    try:
        return float(old_bill) / float(avg_rate)
    except Exception:
        return float(old_bill) / 0.15

def estimate_new_bill(consumption_kwh, retrieved_rows):
    """
    Estimate new bill using the residential rate parsed from retrieved rows.
    If no rows, fall back to default.
    """
    if not retrieved_rows:
        return None, 0.15, 0.0

    row = retrieved_rows[0]
    rate_match = re.search(r"Residential Rate:\s*([0-9.]+)", row)
    if not rate_match:
        # try alternative numeric extraction
        nums = re.findall(r"([0-9]+\.[0-9]+)", row)
        rate = float(nums[0]) if nums else 0.15
    else:
        rate = float(rate_match.group(1))

    fixed = 0.0
    new_bill = consumption_kwh * rate + fixed
    return new_bill, rate, fixed

# ---------------- GPT tips ----------------
def generate_gpt_tips(old_bill, new_bill, usage_kwh, reduction_needed, retrieved_rows, user_info=None):
    context = "\n---\n".join(retrieved_rows) if retrieved_rows else "No utility data found."
    prompt = f"""
You are an energy efficiency assistant. Use the following utility plan context:

{context}

User info:
- Previous bill: ${old_bill}
- Estimated usage: {usage_kwh:.0f} kWh/month
- Target reduction: {reduction_needed:.1f}%
- Additional info: {user_info if user_info else 'None'}

Provide 3-5 actionable energy-saving tips grounded in the retrieved CSV data.
"""
    # Use OpenAI client (chat). Using modern OpenAI client call via `client.chat.completions.create`
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert energy assistant."},
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


# ---------------- Interactive chatbot -----------------
def interactive_gpt(input, context):
    prompt = f"""
You are an energy efficiency assistant. Use the following memory context:

{context}

Based off the chat memory, be as conversational as possible in your reply to the user input below.
User input:
{input}
"""
    # Use OpenAI client (chat). Using modern OpenAI client call via `client.chat.completions.create`
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert energy assistant responding to customer inquiries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    # resp may be an object; convert to dict safely
    try:
        content = resp.choices[0].message.content
    except Exception:
        # fallback for dict-like
        content = resp["choices"][0]["message"]["content"]
    return content.strip()


# ---------------- Main conversational flow ----------------
def run():
    # If you want to ingest, uncomment the next line.
    # Be careful: ingest_csv_to_pinecone will upsert rows each run unless you guard it.
    #ingest_csv_to_pinecone()

    memory = load_memory()

    zip_code = input(f"Enter ZIP code [{memory.get('zip_code','')}]: ") or memory.get("zip_code")
    old_bill = input(f"Previous bill ($) [{memory.get('old_bill','')}]: ") or memory.get("old_bill")
    target_bill = input(f"Target bill ($) [{memory.get('target_bill','')}]: ") or memory.get("target_bill")
    utility_name = input(f"Utility provider name [{memory.get('utility_name','')}]: ") or memory.get("utility_name")
    user_info = input(f"Any usage details (EV, HVAC, Washer/Dryer etc.) [{memory.get('user_info','')}]: ") or memory.get("user_info")

    # ensure numeric
    try:
        old_bill = float(old_bill)
        target_bill = float(target_bill)
    except Exception:
        print("Could not parse bill values. Please enter numeric amounts.")
        return

    # save memory
    memory.update({
        "zip_code": zip_code,
        "old_bill": old_bill,
        "target_bill": target_bill,
        "utility_name": utility_name,
        "user_info": user_info
    })
    save_memory(memory)

    usage_kwh = estimate_usage_from_bill(old_bill)
    query = f"{utility_name} {zip_code} residential rate"
    retrieved_rows = retrieve_csv_rows(query, top_k=3)

    new_bill, rate, fixed = estimate_new_bill(usage_kwh, retrieved_rows)
    if new_bill is None:
        print("Could not estimate new bill (no retrieved utility rates).")
        return

    reduction_needed = (1 - (float(target_bill) / float(new_bill))) * 100

    tips = generate_gpt_tips(old_bill, new_bill, usage_kwh, reduction_needed, retrieved_rows, user_info)

    print(f"\nEstimated new bill: ${new_bill:.2f} (Rate: ${rate:.4f}/kWh)")
    print(f"To reach ${target_bill}, reduce/shift {reduction_needed:.1f}% of usage.")
    print("\n💡 Personalized Energy Tips:\n")
    print(tips)

    #Chatbot response memory file
    chat_memory_file = "chat_memory.txt"

    # ---------------- Chatbot Memory helpers ----------------
    def load_response_memory():
        if os.path.exists(chat_memory_file):
            with open(chat_memory_file, "r") as f:
                return f.read()
        return ""

    def save_response_memory(chat_memory):
        with open(chat_memory_file, "a") as f: #append user input
            f.write(chat_memory + "\n\n")

    load_response_memory()
    save_response_memory(tips)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        chat_memory = load_response_memory()
        response = interactive_gpt(user_input, chat_memory)
        print(f"🤖 Energy Assistant: {response}")

        # Save the conversation to memory
        try:
            save_response_memory(f"You: {user_input}\nEnergy Assistant: {response}")
        except Exception as e:
            continue

if __name__ == "__main__":
    run()
