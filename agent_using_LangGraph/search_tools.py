from dotenv import load_dotenv
import os 
from tavily import TavilyClient

_ = load_dotenv()

client = TavilyClient(apikey=os.environ.get("TAVILY_API_KEY"))

