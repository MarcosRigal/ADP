import os
import pymongo
import datetime
import concurrent.futures
from tqdm import tqdm

# Connect to MongoDB (pymongo connections are thread-safe)
client = pymongo.MongoClient("mongodb://admin:1234@localhost:27017/")
db = client["Milan_CDR_db"]
collection = db["Milan_CDR_c"]

# Directory where .txt files are located
data_directory = "P3/data/"
batch_size = 1000  # Adjust this number based on memory and performance needs

def process_file(filename, position):
    filepath = os.path.join(data_directory, filename)
    
    # Count total lines for a progress bar
    with open(filepath, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    docs = []
    missing_fields_count = 0
    invalid_timestamp_count = 0
    inserted_count = 0

    # Process file with progress bar
    with open(filepath, 'r') as f, tqdm(total=total_lines, desc=filename, position=position, leave=True) as pbar:
        for line in f:
            line = line.strip()
            if not line:
                pbar.update(1)
                continue

            parts = line.split()  # Splits on any whitespace
            if len(parts) < 8:
                missing_fields_count += 1
                pbar.update(1)
                continue

            try:
                square_id = int(parts[0])
                time_interval_str = parts[1]  # Unix timestamp in milliseconds
                country_code = int(parts[2])
                sms_in = float(parts[3])
                sms_out = float(parts[4])
                call_in = float(parts[5])
                call_out = float(parts[6])
                internet = float(parts[7])

                # Convert Unix timestamp to datetime
                try:
                    timestamp_ms = int(time_interval_str)
                    time_interval = datetime.datetime.utcfromtimestamp(timestamp_ms / 1000)
                except ValueError:
                    invalid_timestamp_count += 1
                    pbar.update(1)
                    continue

            except ValueError:
                missing_fields_count += 1
                pbar.update(1)
                continue

            # Build document for MongoDB
            doc = {
                "square_id": square_id,
                "time_interval": time_interval,
                "country_code": country_code,
                "sms_in": sms_in,
                "sms_out": sms_out,
                "call_in": call_in,
                "call_out": call_out,
                "internet": internet
            }
            docs.append(doc)

            # Insert batch when batch size is reached
            if len(docs) >= batch_size:
                try:
                    collection.insert_many(docs, ordered=False)
                    inserted_count += len(docs)
                except Exception as e:
                    print(f"Error inserting batch from {filename}: {e}")
                docs = []

            pbar.update(1)

    # Insert any remaining documents
    if docs:
        try:
            collection.insert_many(docs, ordered=False)
            inserted_count += len(docs)
        except Exception as e:
            print(f"Error inserting final batch from {filename}: {e}")

    # Print summary for this file
    print(f"Finished {filename}: Inserted {inserted_count} docs, Missing fields: {missing_fields_count}, Invalid timestamps: {invalid_timestamp_count}")

def main():
    files = [f for f in os.listdir(data_directory) if f.endswith(".txt")]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        # Assign a unique position for each progress bar
        for pos, filename in enumerate(files):
            futures.append(executor.submit(process_file, filename, pos))
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
