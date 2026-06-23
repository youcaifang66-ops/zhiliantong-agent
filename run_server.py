import os
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["DASHSCOPE_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")

import uvicorn

if __name__ == "__main__":
    uvicorn.run("fastapi_main:app", host="127.0.0.1", port=8001, reload=False)
