from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
engine = create_engine(SUPABASE_DB_URL)