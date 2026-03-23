"""
@File       : chat_tool.py
@Description: Agent 工具集

@Time       : 2026/1/6 10:20
@Author     : hcy18
"""
import httpx
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from app.config.nacos_client import get_nacos_client
from app.utils.logger import app_logger as logger
from app.clients.product_service import get_product_service_client
from app.schemas.product_response_schema import ProductResponseDto
from app.services.rag_service import get_rag_service


# ========== Helper Functions ==========


def _get_tavily_api_key() -> str:
    """从 chat_config 获取 Tavily API Key"""
    chat_config = get_nacos_client().get_chat_config()
    return chat_config.get("tavily_api_key", "")


def _get_hefeng_api_key() -> str:
    """从 chat_config 获取和风天气 API Key"""
    chat_config = get_nacos_client().get_chat_config()
    return chat_config.get("hefeng_weather_api_key", "")


async def _city_lookup(city: str) -> str | None:
    """
    根据城市名称查询 location_id

    Args:
        city: 城市名称，如 "北京"

    Returns:
        location_id 或 None（未找到）
    """
    api_key = _get_hefeng_api_key()
    if not api_key:
        return None

    url = f"https://geo.qweather.com/v2/city/lookup?location={city}&key={api_key}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            data = response.json()
            if data.get("code") == "200" and data.get("location"):
                return data["location"][0].get("id")
    except Exception as e:
        logger.error(f"[城市查询] 查询失败: {e}")
    return None


# ========== Product Tools ==========


@tool
def platform_knowledge_search(query: str) -> str:
    """
    搜索平台规则和知识库，返回检索到的相关文档片段。
    用于回答关于平台政策、规则、流程等问题。
    输入必须是一个具体、清晰的问题。

    Args:
        query: 要搜索的问题，例如 "如何申请退货？"

    Returns:
        检索到的文档片段，或错误信息。
    """
    try:
        rag_service = get_rag_service()
        index = rag_service.get_index()
        logger.info(f"[知识库检索] 问题：{query}")
        retriever = index.as_retriever(similarity_top_k=3)
        docs = retriever.retrieve(query)

        # 格式化返回检索结果
        if docs:
            result_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '未知')
                result_parts.append(f"【文档 {i}】来源：{source}\n{doc.text}")
            result = "\n\n".join(result_parts)
            logger.info(f"[知识库检索] 检索到 {len(docs)} 个相关文档")
            logger.debug(f"[知识库检索结果] {result}")
            return result
        else:
            logger.info(f"[知识库检索] 未找到相关文档")
            return "未在知识库中找到相关信息。"
    except Exception as e:
        logger.error(f"[知识库检索] 查询失败: {e}", exc_info=True)
        return f"查询知识库时发生错误: {str(e)}"


@tool
async def get_new_product(limit: int = 3) -> list[ProductResponseDto]:
    """
    获取最新商品，调用的时候 limit 参数取值 3 ~ 5 , 不建议超过 5，不然消息太长
    Args:
        limit: 限制数，即 获取几个新品
    """
    try:
        logger.info(f"[商品查询] 获取最新商品，limit={limit}")
        product_client = await get_product_service_client()
        result = await product_client.get_new_products(limit=limit)
        logger.info(f"[商品查询] 获取到 {len(result)} 个新品")
        if result:
            product_names = [p.name for p in result]
            logger.debug(f"[商品查询结果] {product_names}")
        return result
    except Exception as e:
        logger.error(f"[商品查询] 获取新品失败: {e}", exc_info=True)
        return []


@tool
async def search_product(query: str, page_number: int = 1, page_size: int = 3) -> list[ProductResponseDto]:
    """
    根据用户对商品的自然语言描述，搜索商品。 注意，但不支持商品字段属性的过滤，仅仅是针对【商品标题】的关键词和语义搜索。
    Args:
        query: 用户查询，比如 拍照好看的手机、苹果笔记本、送女朋友的礼物
        page_number: 分页的页码
        page_size: 一页的大小（不宜过大，不用超过 5，不然消息太长）
    """
    try:
        logger.info(f"[商品查询] 搜索商品，query={query}, page={page_number}, size={page_size}")
        product_client = await get_product_service_client()
        result = await product_client.search_products(query, page_number, page_size)
        logger.info(f"[商品查询] 搜索到 {len(result)} 个商品")
        if result:
            product_names = [p.name for p in result]
            logger.debug(f"[商品查询结果] {product_names}")
        return result
    except Exception as e:
        logger.error(f"[商品查询] 搜索商品失败: {e}", exc_info=True)
        return []


@tool
async def get_product_detail(product_id: int) -> ProductResponseDto | None:
    """
    根据商品ID查询商品详情，用于获取商品的完整信息包括价格、描述、库存、款式等，如果返回为空，说明商品详情获取失败！
    Args:
        product_id: 商品ID
    """
    try:
        logger.info(f"[商品查询] 获取商品详情，product_id={product_id}")
        product_client = await get_product_service_client()
        result = await product_client.get_product_by_id(product_id)
        logger.info(f"[商品查询] 获取商品详情成功！product_name={result.name}")
        return result
    except Exception as e:
        logger.error(f"[商品查询] 获取商品详情失败: {e}", exc_info=True)
        return None


# ========== External Tools ==========


@tool
def tavily_search(query: str) -> str:
    """
    Tavily 联网搜索工具，用于搜索网络上最新信息和回答实时问题。

    Args:
        query: 搜索关键词或问题，例如 "今天有什么科技新闻"

    Returns:
        搜索结果摘要，包含标题、URL 和内容摘要
    """
    api_key = _get_tavily_api_key()
    if not api_key:
        return "Tavily API Key 未配置，请在 chat_config 中配置 tavily_api_key"

    try:
        logger.info(f"[Tavily搜索] query: {query}")
        tavily = TavilySearch(max_results=3, tavily_api_key=api_key)
        result = tavily.invoke(query)
        logger.info(f"[Tavily搜索] 成功")
        return result
    except Exception as e:
        logger.error(f"[Tavily搜索] 失败: {e}", exc_info=True)
        return f"搜索失败: {str(e)}"


@tool
async def get_current_weather(city: str) -> str:
    """
    查询城市实时天气。

    Args:
        city: 城市名称，例如 "北京"、"上海"、"广州"

    Returns:
        格式化后的天气信息，包含温度、湿度、风力等
    """
    api_key = _get_hefeng_api_key()
    if not api_key:
        return "和风天气 API Key 未配置，请在 chat_config 中配置 hefeng_weather_api_key"

    try:
        logger.info(f"[实时天气] 查询城市: {city}")

        # Step 1: 城市 lookup 获取 location_id
        location_id = await _city_lookup(city)
        if not location_id:
            logger.warning(f"[实时天气] 城市未找到: {city}")
            return f"未找到城市 '{city}'，请确认城市名称是否正确"

        # Step 2: 查询实时天气
        url = f"https://weather.qweather.com/v7/weather/now?location={location_id}&key={api_key}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            data = response.json()

        if data.get("code") != "200":
            return f"查询天气失败: {data.get('msg', '未知错误')}"

        weather = data.get("now", {})
        result = (
            f"【{city}实时天气】\n"
            f"- 温度: {weather.get('temp', 'N/A')}°C\n"
            f"- 体感温度: {weather.get('feelsLike', 'N/A')}°C\n"
            f"- 天气: {weather.get('text', 'N/A')}\n"
            f"- 风速: {weather.get('windSpeed', 'N/A')} km/h（{weather.get('windDir', 'N/A')}）\n"
            f"- 湿度: {weather.get('humidity', 'N/A')}%\n"
            f"- 能见度: {weather.get('vis', 'N/A')} km\n"
            f"- 气压: {weather.get('pressure', 'N/A')} hPa\n"
            f"- 更新时间: {data.get('updateTime', 'N/A')}"
        )
        logger.info(f"[实时天气] 成功: {city}")
        return result

    except Exception as e:
        logger.error(f"[实时天气] 失败: {e}", exc_info=True)
        return f"查询天气失败: {str(e)}"


@tool
async def get_forecast_weather(city: str, days: int = 3) -> str:
    """
    查询城市天气预报，支持未来 3-30 天。

    Args:
        city: 城市名称，例如 "北京"、"上海"
        days: 预报天数，取值范围 3-30，默认为 3

    Returns:
        格式化后的天气预报信息
    """
    api_key = _get_hefeng_api_key()
    if not api_key:
        return "和风天气 API Key 未配置，请在 chat_config 中配置 hefeng_weather_api_key"

    if days < 3 or days > 30:
        return "预报天数必须在 3-30 天之间"

    try:
        logger.info(f"[天气预报] 查询城市: {city}, 天数: {days}")

        # Step 1: 城市 lookup 获取 location_id
        location_id = await _city_lookup(city)
        if not location_id:
            logger.warning(f"[天气预报] 城市未找到: {city}")
            return f"未找到城市 '{city}'，请确认城市名称是否正确"

        # Step 2: 查询天气预报
        url = f"https://weather.qweather.com/v7/weather/{days}d?location={location_id}&key={api_key}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            data = response.json()

        if data.get("code") != "200":
            return f"查询天气失败: {data.get('msg', '未知错误')}"

        daily_list = data.get("daily", [])
        if not daily_list:
            return f"未找到 {city} 的天气预报数据"

        # 格式化输出
        result_parts = [f"【{city}天气预报】（共 {len(daily_list)} 天）\n"]
        for i, day in enumerate(daily_list, 1):
            result_parts.append(
                f"\n📅 第 {i} 天 ({day.get('fxDate', 'N/A')})\n"
                f"   天气: {day.get('textDay', 'N/A')} → {day.get('textNight', 'N/A')}\n"
                f"   温度: {day.get('tempMin', 'N/A')}°C ~ {day.get('tempMax', 'N/A')}°C\n"
                f"   降水概率: {day.get('pop', 'N/A')}%\n"
                f"   风速: {day.get('windSpeedDay', 'N/A')} km/h（{day.get('windDirDay', 'N/A')}）"
            )

        result = "\n".join(result_parts)
        logger.info(f"[天气预报] 成功: {city}, {len(daily_list)} 天")
        return result

    except Exception as e:
        logger.error(f"[天气预报] 失败: {e}", exc_info=True)
        return f"查询天气失败: {str(e)}"