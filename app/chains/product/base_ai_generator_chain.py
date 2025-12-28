"""
@File       : base_ai_generator_chain.py
@Description:

@Time       : 2025/12/28 18:56
@Author     : hcy18
"""
import asyncio
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any

from langchain_core.output_parsers import PydanticOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.services.llm_service import get_llm_service
from app.utils.image_util import load_image_from_url, is_base64_image
from app.utils.logger import app_logger as logger

# 泛型变量：Input = 子类的输入数据结构，Output = 子类的返回 Pydantic 模型
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class VisionAwareAIGenerator(ABC, Generic[InputType, OutputType]):
    """支持图文输入的 AI 生成器抽象基类"""

    def __init__(self):
        self.llm_service = get_llm_service()

    @abstractmethod
    def _has_images(self, input_data: InputType) -> bool:
        """判断输入是否包含有效图片 URL 列表"""
        pass

    def _get_text_only_system_prompt(self) -> str:
        """纯文本模式的系统提示"""
        return ""

    def _get_vision_system_prompt(self) -> str:
        """视觉模式的系统提示"""
        return ""

    def _build_text_only_human_message_content(self, input_data: InputType) -> str:
        """构建纯文本 HumanMessage 的 content（字符串）"""
        return ""

    def _build_vision_human_message_base_text(self, input_data: InputType) -> str:
        """构建视觉 HumanMessage 中的文本部分（用于拼接图片）"""
        return ""

    @abstractmethod
    def _get_output_parser(self) -> BaseOutputParser:
        """返回 输出解析器"""
        pass


    async def _generate_with_text_only(self, input_data: InputType) -> OutputType:
        """使用纯文本进行生成"""

        system_prompt = (
            self._get_text_only_system_prompt().strip() + "\n\n"
            "请严格按照以下格式输出结果，不要包含任何额外解释或文本：\n"
            "{format_instructions}"
        )
        human_content = self._build_text_only_human_message_content(input_data)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_content)
        ])

        parser = self._get_output_parser()
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        llm = self.llm_service.get_chat_model()
        chain = prompt | llm | parser
        return await chain.ainvoke({})


    async def _generate_with_vision(self, input_data: InputType) -> OutputType:
        """使用视觉模型进行生成"""
        system_prompt = (
            self._get_vision_system_prompt().strip() + "\n\n"
            "请严格按照以下格式输出结果，不要包含任何额外解释或文本：\n"
            "{format_instructions}"
        )

        # 从输入泛型里获取图片 URL 列表（子类需保证此方法可用）
        image_refs = self._extract_image_urls(input_data)

        # 2. 分离 base64 和普通 URL
        base64_images: list[str] = []
        url_images: list[str] = []

        for ref in image_refs:
            if is_base64_image(ref):
                base64_images.append(ref)
            else:
                url_images.append(ref)

        # 3. 仅对 URL 图片并发加载
        url_tasks = [load_image_from_url(url) for url in url_images]
        loaded_base64_from_urls = await asyncio.gather(*url_tasks) if url_tasks else []

        # 4. 合并：base64 直接保留，URL 转为 base64 后追加
        all_base64_images = base64_images + list(loaded_base64_from_urls)

        base_text = self._build_vision_human_message_base_text(input_data)
        content = [{"type": "text", "text": base_text}]
        for base64_str in all_base64_images:
            content.append({"type": "image_url", "image_url": {"url": base64_str}})

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", content)
        ])

        # 5, 输出解析器
        parser = self._get_output_parser()
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        # 6, 链与执行
        vision_model = self.llm_service.get_vision_model()
        chain = prompt | vision_model | parser
        return await chain.ainvoke({})


    def _extract_image_urls(self, input_data: InputType) -> list[str]:
        """默认从 input_data.image_urls 提取 image url，子类可重写"""
        if hasattr(input_data, 'image_urls') and isinstance(input_data.image_urls, list):
            return [url for url in input_data.image_urls if url]
        return []


    async def generate(self, input_data: InputType) -> OutputType:
        """统一入口：自动根据是否有图片选择生成方式"""
        if self._has_images(input_data):
            return await self._generate_with_vision(input_data)
        else:
            return await self._generate_with_text_only(input_data)