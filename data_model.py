from pydantic import BaseModel
from typing import List

class ImageTagRequest(BaseModel):
    image_url: str

class ImageTagRequestBase64(BaseModel):
    image: str    
    suffix: str

class ImageTagResponse(BaseModel):
    tags: list