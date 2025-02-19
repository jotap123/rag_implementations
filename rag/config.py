from pathlib import Path
from datetime import datetime

power_chat = "gpt-4o-mini"
chat = "microsoft/Phi-3.5-mini-instruct"
coder = "meta-llama/Llama-3.2-3B-Instruct"

TODAY = datetime.today().date().strftime("%Y-%m-%d")


def get_root_dir():
    cur_dir = Path(__file__).resolve().parent

    while cur_dir.name != 'rag_implementations':
        cur_dir = cur_dir.parent
    return str(cur_dir)
