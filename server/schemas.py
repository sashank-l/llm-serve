"""Pydantic schemas for OpenAI-compatible API (v1)."""
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool]       = False
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float]       = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int]         = Field(default=-1)
    max_tokens: Optional[int]    = Field(default=256, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str]          = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int     = 0
    completion_tokens: int = 0
    total_tokens: int      = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str             = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
