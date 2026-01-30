"""
@File       : search_keyword_chain.py
@Description:

@Time       : 2026/1/4 5:32
@Author     : hcy18
"""
from typing import Optional

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser

from app.chains.product.base_ai_generator_chain import VisionAwareAIGenerator, InputType
from app.schemas.search_schema import SearchKeywordEnhanceResponse, SearchKeyWordEnhanceRequest


class SearchKeywordEnhanceChain(VisionAwareAIGenerator[SearchKeyWordEnhanceRequest, SearchKeywordEnhanceResponse]):
    """搜索词增强chain"""

    _instance: Optional["SearchKeywordEnhanceChain"] = None

    def _has_images(self, input_data: InputType) -> bool:
        return False

    def _get_output_parser(self) -> BaseOutputParser:
        return PydanticOutputParser(pydantic_object=SearchKeywordEnhanceResponse)

    def _get_text_only_system_prompt(self) -> str:
        return """
              你是一个专业的**电商搜索理解助手**。请将用户的自然语言搜索的关键词，拆解为两类关键词：

              1. 【核心词】：指用户要找的**具体商品品类或实体**（如"手机""书籍""羽绒服""蓝牙耳机""办公椅"）。  
                 - 必须是名词，且代表商品本身。
                 - 也可以是品牌名称（如果有的话），如 华为、苹果、iPhone、爱马仕、戴尔
                 - 如果用户未明确品类（如"送女友的礼物"），则留空，去做扩展词处理。
                 - 通常只有 1 个，但最多不超过 2 个。核心词是必然在原文包含的。
                 - 如果用户只输入了形容词，没有实体物品，这是非常极端的情况，那么你把形容词识别出来作为名词，比如 "可爱"、"舒服"  

              2. 【扩展词】：包括以下任意类型：
                 - 形容词（如"保暖""轻薄""高清"）
                 - 使用场景（如"冬季""打游戏""上班通勤""礼物""礼品""中秋节""节日""过年"）
                 - 功能卖点（如"续航""像素""可折叠"）
                 - 同义/近义表达（如"裤子"→"长裤"，"拍照好"→"影像清晰"，"旗舰"->"旗舰店"）
                 - 季节/人群/风格（如"冬季""男士""学生""复古""女友"）

              3. 特别注意，有些形容词要尽量扩展出名词。比如 用户搜索 "续航时间长的手机", 那么 核心词是 ["手机"], 扩展词是 ["续航", "续航时间长", "长续航"]；用户搜 "拍照好看的手机"，扩展词有"拍照"
              
              4. 注意错别字，用户输入的搜索关键词，可能有错别字，你要能识别并纠正！
              
              5. 【品类映射规则 - 重要】：对于带性别/人群/季节前缀的品类词，请提取出**基础品类词**作为核心词：
                 - 衣服类，如果带“子”，则“子”字去掉，更符合商家术语
                 - 女鞋/男鞋/童鞋/女式鞋/男士鞋 → 核心词 [鞋]， 而非“鞋子”
                 - 女装/男装/童装 → 核心词 [服装] 或 [衣服]
                 - 女包/男包/童包 → 核心词 [包]
                 - 女裙/女式裙子 → 核心词 [裙]，而非“裙子”
                 - 女裤/男裤/女式裤/男士裤 → 核心词 [裤]，而非“裤子”
                 - 女T恤/男T恤/女式T恤/男士T恤 → 核心词 [T恤]
                 - 女外套/男外套 → 核心词 [外套]
                 - 女上衣/男上衣/女式上衣/男士上衣 → 核心词 [上衣]
                 - 女卫衣/男卫衣/女式卫衣/男士卫衣 → 核心词 [卫衣]

                 示例：
                 输入：女鞋  → 核心词 [鞋]
                 输入：男装  → 核心词 [服装]
                 输入：女式裙子 → 核心词 [裙]
                 输入：女式阔腿裤高腰 → 核心词 [裤]

              输出格式参考如下：
              core_words: [核心词1, 核心词2]
              expand_words: [扩展词1, 扩展词2, 扩展词3, ...]

              示例：
              输入：冬季穿的保暖加绒裤子  
              输出：  
              core_words: [裤]  # 裤是常用词
              expand_words: [冬季, 保暖, 加绒, 厚实, 防寒]

              （说明：因为搜索词是学习相关的，因此可能是 书籍、教程、视频教学 等词汇。）
              输入：Python 入门
              输出：  
              core_words: [书籍]  
              expand_words: [python, 入门, 新手, 教程]


              输入：拍照好看的手机  
              输出：  
              core_words: [手机]  
              expand_words: [像素, 拍照, 影像, 清晰, 长焦, 旗舰店]

              输入：打游戏不卡的高性能手机  
              输出：  
              core_words: [手机]  
              expand_words: [打游戏, 高性能, 不卡, 高帧率, 散热好, 旗舰店]

              输入：送女朋友的生日礼物  
              输出：  
              core_words: []  
              expand_words: [送女友, 生日, 礼物, 精美, 浪漫, 实用]

              输入：苹果手机  
              输出：  
              core_words: [手机]  
              expand_words: [苹果, iPhone, 智能手机, 旗舰店]

              输入：女鞋  
              输出：  
              core_words: [鞋子]  
              expand_words: [女士, 女性, 高跟, 平底, 休闲, 正式, 春夏, 秋冬]

              输入：女式阔腿裤高腰  
              输出：  
              core_words: [裤子]  
              expand_words: [女式, 女士, 女性, 阔腿, 高腰, 修身]
        """

    def _build_text_only_human_message_content(self, input_data: SearchKeyWordEnhanceRequest) -> str:
        return f"用户搜索的关键词是: {input_data.keyword}"


    @classmethod
    def get_instance(cls) -> "SearchKeywordEnhanceChain":
        """
        无状态 chain，单例 ProductAuditChain.

        Returns:
            ProductAuditChain instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


