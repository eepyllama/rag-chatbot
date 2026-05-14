from dotenv import load_dotenv
import os

load_dotenv()

print("Pinecode key loaded: ", bool(os.getenv("PINECONE_API_KEY")))
print("Gro key loaded: ", bool(os.getenv("GROQ_API_KEY")))
print("Setup completed")