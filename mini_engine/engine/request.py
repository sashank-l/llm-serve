"""
engine/request.py
Request, SamplingParams, and RequestStatus dataclasses.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import time


class RequestStatus(Enum):
    WAITING   = "waiting"
    RUNNING   = "running"
    FINISHED  = "finished"
    CANCELLED = "cancelled"
    FAILED    = "failed"


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    max_new_tokens: int = 256
    stop_strings: List[str] = field(default_factory=list)
    stream: bool = True
    seed: Optional[int] = None  # Seed for reproducible random sampling


@dataclass
class Request:
    request_id: str
    prompt_tokens: List[int]
    sampling_params: SamplingParams
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: List[int] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    arrival_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    output_queue: Optional[object] = None
