"""
@File       : search_schema.py
@Description:

@Time       : 2026/1/4 5:28
@Author     : hcy18
"""
from typing import Optional

from app.schemas import CamelCaseModel
from pydantic import Field


class SearchKeyWordEnhanceRequest(CamelCaseModel):
    keyword: str = Field(..., description="关键词")

class SearchKeywordEnhanceResponse(CamelCaseModel):
    """对用户输入的搜索词进行扩展和，并识别出核心词和扩展词"""

    core_words: list[str] = Field(..., description="搜索用的核心词")
    expand_words: Optional[list[str]] = Field(default=None, description="搜索用的扩展词")