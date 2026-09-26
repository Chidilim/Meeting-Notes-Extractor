import datetime
from typing import Dict, Literal
from enum import Enum

from pydantic import BaseModel 

class MeeetingNote(BaseModel):
   meeting: Dict

class ChunkStratCater(Enum):
    FIXED = 0
    TOPIC = 1
    ITEM = 2
    TOPIC_ITEM = 3
    

class ChunkStrat(BaseModel):
    chunk_strategy: ChunkStratCater

class ExtractedItem(BaseModel):
    type: Literal["action_item", "decision"]
    text: str
    owner: str | None = None

class ExtractedItems(BaseModel):
    items: list[ExtractedItem]
    
class query(BaseModel):
    query: str