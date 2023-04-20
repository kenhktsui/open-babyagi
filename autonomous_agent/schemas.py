from typing import Optional
from pydantic import BaseModel


class Task(BaseModel):
    task_id: int
    task_name: str
    result_id: Optional[str] = None
    result: Optional[str] = None
