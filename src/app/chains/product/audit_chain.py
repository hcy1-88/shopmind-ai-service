"""
@File       : audit_chain.py
@Description:

@Time       : 2025/12/28 21:41
@Author     : hcy18
"""
import asyncio
from typing import Optional
from urllib.parse import urlparse

from fastapi.exceptions import RequestValidationError
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.chains.product.base_ai_generator_chain import VisionAwareAIGenerator
from app.schemas.product_audit import (
    ProductAuditRequest,
    ProductAuditResponse,
    CheckResult,
    ImageCheckResult,
)
from app.utils.image_util import load_image_from_url
from app.utils.logger import app_logger as logger


class ProductAuditChain(VisionAwareAIGenerator[ProductAuditRequest, ProductAuditResponse]):
    """商品AI审核 chain"""

    _instance: Optional["ProductAuditChain"] = None

    def _has_images(self, input_data: ProductAuditRequest) -> bool:
        """验证图片参数"""
        if not input_data.cover_image:
            raise RequestValidationError(f"商品没有封面图，商品 id：{input_data.product_id}")
        elif not input_data.detail_images or len(input_data.detail_images) < 1:
            raise RequestValidationError(f"商品详情图至少需要1张，商品 id：{input_data.product_id}")
        
        # 验证封面图必须是 HTTP/HTTPS URL
        cover_parsed = urlparse(input_data.cover_image)
        if not cover_parsed.scheme or cover_parsed.scheme.lower() not in ('http', 'https'):
            raise RequestValidationError(
                f"商品封面图必须是 HTTP/HTTPS URL 格式，商品 id：{input_data.product_id}，当前值：{input_data.cover_image[:100]}"
            )
        
        # 验证所有详情图必须是 HTTP/HTTPS URL
        for idx, detail_img in enumerate(input_data.detail_images):
            detail_parsed = urlparse(detail_img)
            if not detail_parsed.scheme or detail_parsed.scheme.lower() not in ('http', 'https'):
                raise RequestValidationError(
                    f"商品详情图第 {idx + 1} 张必须是 HTTP/HTTPS URL 格式，商品 id：{input_data.product_id}，当前值：{detail_img[:100]}"
                )
        
        return True

    def _get_output_parser(self) -> BaseOutputParser:
        return PydanticOutputParser(pydantic_object=ProductAuditResponse)

    def _extract_image_urls(self, input_data: ProductAuditRequest) -> list[str]:
        """提取所有图片 URL：封面图 + 详情图"""
        return [input_data.cover_image] + input_data.detail_images

    async def _generate_with_vision(self, input_data: ProductAuditRequest) -> ProductAuditResponse:
        """重写视觉模式生成方法，分步骤审核"""
        try:
            # 第一步：审核标题和描述
            title_check_result = await self._audit_text_content(input_data)
            logger.info(f"商品 {input_data.product_id} 文本审核完成")

            # 第二步：逐张审核图片
            image_check_results = await self._audit_images(input_data)
            logger.info(f"商品 {input_data.product_id} 图片审核完成，共 {len(image_check_results)} 张")

            # 第三步：综合判断审核状态，全部通过则通过，有一个拒绝则拒绝
            all_passed = title_check_result.passed and all(
                img_result.passed for img_result in image_check_results
            )
            audit_status = "approved" if all_passed else "rejected"

            # 第四步：生成拒绝原因和建议
            reject_reason = None
            suggestions = []
            
            if not all_passed:
                reject_reasons = []
                if not title_check_result.passed:
                    reject_reasons.append(f"标题问题：{title_check_result.detail}")
                    suggestions.append("请修改商品标题")
                
                failed_images = [
                    img for img in image_check_results if not img.passed
                ]
                if failed_images:
                    reject_reasons.append(
                        f"图片问题：共 {len(failed_images)} 张图片不合规"
                    )
                    suggestions.append("请替换不合规的商品图片")
                
                reject_reason = "；".join(reject_reasons)

            return ProductAuditResponse(
                audit_status=audit_status,
                title_check_result=title_check_result,
                image_check_results=image_check_results,
                reject_reason=reject_reason,
                suggestions=suggestions,
            )

        except Exception as e:
            logger.error(f"商品审核失败，商品 id：{input_data.product_id}，错误：{e}", exc_info=True)
            # 返回默认拒绝结果
            return ProductAuditResponse(
                audit_status="rejected",
                title_check_result=CheckResult(
                    passed=False,
                    issue_types=["system_error"],
                    confidence=1.0,
                    detail=f"审核系统错误: {str(e)}",
                ),
                image_check_results=[],
                reject_reason=f"系统错误: {str(e)}",
                suggestions=["请稍后重试"],
            )

    async def _audit_text_content(self, input_data: ProductAuditRequest) -> CheckResult:
        """审核标题和描述文本内容"""
        system_prompt = """你是一个专业的电商商品文本内容审核专家。请审核商品的标题和描述是否合规。

【重要原则】审核规则应宽松，只有在明确、确定违规的情况下才拒绝。对于"可能"、"疑似"、"存疑"的情况，应倾向于通过。

审核标准（仅在明确违规时拒绝）：
1. **明确违规内容**：明确包含色情、暴力、政治敏感、违法不良信息（存疑或可能的情况应通过）
2. **明确极限词**：明确使用"全网最低"、"史上最强"、"100%有效"等绝对化用语（轻微夸大或可能的情况应通过）
3. **明确联系方式**：明确包含电话、微信、QQ、网址等站外引导信息（模糊或可能的情况应通过）
4. **明确违禁品**：明确涉及枪支、毒品、赌博、成人用品等（存疑的情况应通过）
5. **允许合理营销**：可使用"政府补贴"、"限时优惠"、"热销"等常见营销用语

【判断标准】
- 只有在置信度 >= 0.8 且明确违规时，才设置 passed=false
- 对于"可能"、"疑似"、"存疑"、"不确定"的情况，应设置 passed=true
- 轻微夸大、模糊表述、边界情况应通过审核

请返回审核结果，包含：
- passed: 是否通过（true/false，存疑时倾向于true）
- issue_types: 问题类型列表，如 ["sensitive_words", "illegal_content", "contact_info"]，若通过则为空数组
- confidence: 置信度（0-1），只有置信度>=0.8且明确违规时才拒绝
- detail: 详细说明"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\n请严格按照以下格式输出结果：\n{format_instructions}"),
            ("human", "商品标题：{title}\n\n商品描述：{description}")
        ])

        parser = PydanticOutputParser(pydantic_object=CheckResult)
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        llm = self.llm_service.get_chat_model()
        chain = prompt | llm | parser

        return await chain.ainvoke({
            "title": input_data.title,
            "description": input_data.description,
        })

    async def _audit_images(self, input_data: ProductAuditRequest) -> list[ImageCheckResult]:
        """逐张审核所有图片（封面 + 详情）"""
        # 收集所有图片 URL（原始 URL，用于结果中）
        all_image_urls = [input_data.cover_image] + input_data.detail_images
        
        logger.info(
            f"开始审核商品 {input_data.product_id} 的图片，"
            f"共 {len(all_image_urls)} 张（1张封面 + {len(input_data.detail_images)}张详情）"
        )
        
        # 并发审核所有图片
        tasks = [
            self._audit_single_image(url, idx, len(all_image_urls))
            for idx, url in enumerate(all_image_urls)
        ]
        results = await asyncio.gather(*tasks)
        
        logger.info(
            f"商品 {input_data.product_id} 图片审核完成，"
            f"通过：{sum(1 for r in results if r.passed)}/{len(results)}"
        )
        
        return results

    async def _audit_single_image(
        self, 
        image_url: str, 
        index: int, 
        total: int
    ) -> ImageCheckResult:
        """审核单张图片"""
        image_type = "封面图" if index == 0 else f"详情图{index}"
        
        # 验证图片 URL 必须是 HTTP/HTTPS 格式
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or parsed_url.scheme.lower() not in ('http', 'https'):
            raise RequestValidationError(
                f"商品{image_type}必须是 HTTP/HTTPS URL 格式，当前值：{image_url[:100]}"
            )
        
        try:
            logger.debug(f"开始审核{image_type}（{index + 1}/{total}）：{image_url[:100]}")
            
            # 1. 从 HTTP URL 加载图片为 base64
            image_base64 = await load_image_from_url(image_url)

            # 2. 构建审核提示
            system_prompt = """你是一个专业的电商平台图片审核专家。请审核这张商品图片是否符合平台规范。

【重要原则】审核规则应宽松，只有在明确、确定违规的情况下才拒绝。对于"可能"、"疑似"、"存疑"的情况，应倾向于通过。

审核标准（仅在明确违规时拒绝）：
1. **明确色情、暴力、血腥内容**：明确包含色情、暴力、血腥内容（存疑或可能的情况应通过）
2. **明确政治敏感内容**：明确包含政治敏感内容（模糊或可能的情况应通过）
3. **明确虚假宣传**：明确包含虚假宣传或严重误导性内容（轻微夸大应通过）
4. **明确站外引导**：明确包含二维码、联系方式等站外引导信息（模糊或可能的情况应通过）
5. **允许正常展示**：图片清晰、真实展示商品即可通过

【判断标准】
- 只有在置信度 >= 0.8 且明确违规时，才设置 passed=false
- 对于"可能"、"疑似"、"存疑"、"不确定"的情况，应设置 passed=true
- 边界情况、模糊内容、轻微问题应通过审核
- 正常商品展示、营销图片应通过

请返回审核结果，包含：
- passed: 是否通过（true/false，存疑时倾向于true）
- issue_types: 问题类型列表，如 ["violence", "adult_content", "qr_code"]，若通过则为空数组
- confidence: 置信度（0-1），只有置信度>=0.8且明确违规时才拒绝
- detail: 详细说明"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\n请严格按照以下格式输出结果：\n{format_instructions}"),
                ("human", [
                    {"type": "text", "text": f"请审核这张商品图片（{image_type}，第 {index + 1}/{total} 张）："},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ])
            ])

            # 3. 使用不包含 image_url 字段的临时模型
            class TempImageCheckResult(CheckResult):
                """临时图片审核结果（不含 image_url）"""
                pass

            parser = PydanticOutputParser(pydantic_object=TempImageCheckResult)
            prompt = prompt.partial(format_instructions=parser.get_format_instructions())

            vision_model = self.llm_service.get_vision_model()
            chain = prompt | vision_model | parser

            # 4. 执行审核
            temp_result = await chain.ainvoke({})

            # 5. 构建完整结果，手动添加 image_url
            result = ImageCheckResult(
                image_url=image_url,  # 使用原始 URL
                passed=temp_result.passed,
                issue_types=temp_result.issue_types,
                confidence=temp_result.confidence,
                detail=temp_result.detail,
            )
            
            logger.debug(
                f"{image_type}审核完成：{'通过' if result.passed else '不通过'}，"
                f"置信度：{result.confidence}"
            )
            
            return result

        except Exception as e:
            logger.error(f"{image_type}审核失败，URL: {image_url[:100]}，错误：{e}", exc_info=True)
            # 返回失败结果
            return ImageCheckResult(
                image_url=image_url,
                passed=False,
                issue_types=["system_error"],
                confidence=0.0,
                detail=f"图片审核失败: {str(e)}",
            )

    @classmethod
    def get_instance(cls) -> "ProductAuditChain":
        """
        无状态 chain，单例 ProductAuditChain.

        Returns:
            ProductAuditChain instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
