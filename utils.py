import os
import requests

from dotenv import load_dotenv
load_dotenv()


TLG_TOKEN = os.getenv("TLG_TOKEN")
TLG_CHAT_ID = os.getenv("TLG_CHAT_ID")


def to_tlg(msg):
    import requests
    url = f"https://api.telegram.org/bot279783998:{TLG_TOKEN}/sendMessage?chat_id={TLG_CHAT_ID}&text={msg}"
    try:
        res = requests.get(url)
        return res.status_code
    except Exception as e:
        return str(e)