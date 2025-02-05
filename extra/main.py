from fastapi import FastAPI, HTTPException
# from typing import Optional
from pydantic import BaseModel

from main1 import get_json


app = FastAPI()

class Item(BaseModel):
    link: str

@app.post("/a")
async def getdata(data : Item):
    response = get_json(data.link)
    return response


@app.get("/abc/{url}")
async def postdata(url):
    return {url:"is a test"}