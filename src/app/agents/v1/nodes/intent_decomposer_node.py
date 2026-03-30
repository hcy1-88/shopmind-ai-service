"""
@File       : intent_decomposer_node.py
@Description: 意图分解节点 - 负责把用户的 query 分解成 subtask 列表，并提取槽位信息

@Time       : 2026/3/8 21:42
@Author     : hcy18
"""
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from app.agents.v1.schema import (
    ShopmindAssistantContext,
    ShopmindAgentState,
    IntentResponse,
    IntentItem,
    SubTask,
    ShoppingSubTask,
    PlatformSubTask,
    ChitchatSubTask,
    ComparisonSubTask,
    IntentCategory,
    TaskStatus,
)
from app.utils.logger import app_logger as logger


# 活跃任务状态
ACTIVE_TASK_STATUSES = {TaskStatus.NEW, TaskStatus.CLARIFYING, TaskStatus.READY}


def _filter_active_subtasks(subtasks: list[SubTask], max_count) -> list[SubTask]:
    """过滤并限制活跃的历史子任务

    1. 过滤掉已完成的任务（COMPLETED/FAILED）
    2. 限制只发送最近 max_count 个任务
    """
    # 过滤出活跃任务
    active_tasks = [t for t in subtasks if t.status in ACTIVE_TASK_STATUSES]
    # 按时间倒序，取最近 max_count 个
    return active_tasks[-max_count:]


async def intent_decomposer_node(state: ShopmindAgentState, runtime: Runtime[ShopmindAssistantContext]):
    """意图分解器节点，负责把用户的 query 分解成 subtask 列表"""
    context = runtime.context
    logger.info(f"[intent_decomposer_node] thread_id: {context.thread_id}")

    llm = context.reasoning_llm
    max_clarification_count = context.max_clarification_count
    max_history_task_count = context.max_history_task_count
    max_search_loop = context.max_search_loop

    subtasks = state.get("sub_tasks", [])
    filtered_subtasks = _filter_active_subtasks(subtasks, max_count=max_history_task_count)
    intent_resp = await intent_analyze(llm, state.get("rewritten_query", ""), filtered_subtasks)

    # 2. 处理每个意图项
    for intent_item in intent_resp.intent_items:
        logger.info(f"【意图识别结果】- {intent_item}")
        # 判断是新的意图还是旧的意图
        is_new, matched_subtask = is_new_intent(intent_item, subtasks)
        current_subtask = matched_subtask

        if intent_item.intent == IntentCategory.SHOPPING:
            if is_new:
                # 创建新的 ShoppingSubTask 并填充槽位
                current_subtask = ShoppingSubTask(
                    category=IntentCategory.SHOPPING,
                    sub_query=intent_item.sub_query,
                    status=TaskStatus.NEW,
                    product_category=intent_item.extracted_slots.get("product_category"),
                    keywords=intent_item.extracted_slots.get("keywords", []),
                    filters=intent_item.extracted_slots.get("filters", {}),
                    is_replace_products=intent_item.is_replace_products,
                    max_search_loop=max_search_loop
                )
                subtasks.append(current_subtask)
            else:
                # 更新已有的 SubTask 槽位
                _update_subtask_slots(matched_subtask, intent_item.extracted_slots)
                # 更新 is_replace_products
                matched_subtask.is_replace_products = intent_item.is_replace_products
                current_subtask = matched_subtask
            # 判断是否需要澄清
            _handle_clarification(current_subtask, intent_item, max_clarification_count)

        elif intent_item.intent == IntentCategory.PLATFORM:
            if is_new:
                # 如果是平台规则，创建 PlatformSubTask
                new_subtask = PlatformSubTask(
                    category=IntentCategory.PLATFORM,
                    sub_query=intent_item.sub_query,
                    status=TaskStatus.NEW,
                )
                current_subtask = new_subtask
                subtasks.append(new_subtask)

        elif intent_item.intent == IntentCategory.CHITCHAT:
            if is_new:
                # 创建 ChitchatSubTask
                new_subtask = ChitchatSubTask(
                    category=IntentCategory.CHITCHAT,
                    sub_query=intent_item.sub_query,
                    status=TaskStatus.NEW,
                )
                current_subtask = new_subtask
                subtasks.append(new_subtask)

        elif intent_item.intent == IntentCategory.COMPARISON:
            if is_new:
                # 创建 ComparisonSubTask，从 matched_task_id 获取 has_recommended_product_ids
                product_ids = []
                if intent_item.matched_task_id:
                    for t in subtasks:
                        if t.task_id == intent_item.matched_task_id:
                            if isinstance(t, ShoppingSubTask):
                                product_ids = t.has_recommended_product_ids or []
                            break

                new_subtask = ComparisonSubTask(
                    category=IntentCategory.COMPARISON,
                    sub_query=intent_item.sub_query,
                    status=TaskStatus.NEW,
                    product_ids=product_ids,
                )
                current_subtask = new_subtask
                subtasks.append(new_subtask)

        # 加入本轮的目标任务
        state["current_tasks"].append(current_subtask)

    # 3. 更新 state
    state["sub_tasks"] = subtasks
    return state


def _update_subtask_slots(subtask: SubTask, extracted_slots: dict):
    """更新已有 SubTask 的槽位 - LLM 返回的是完整的全量信息，直接赋值"""
    if not isinstance(subtask, ShoppingSubTask):
        return

    # 更新品类
    if extracted_slots.get("product_category") and not subtask.product_category:
        subtask.product_category = extracted_slots.get("product_category")

    # 直接赋值 - LLM 返回的是信息完整的 keywords
    new_keywords = extracted_slots.get("keywords", [])
    if new_keywords:
        subtask.keywords = new_keywords

    # 直接赋值 - LLM 返回的是完整的 filters
    subtask.filters = extracted_slots.get("filters", {})


def _handle_clarification(subtask: SubTask, intent_item: IntentItem, max_clarification_count: int):
    """判断是否需要澄清，并处理澄清逻辑

    判断依据：
    1. explicit_search=true → 立即触发搜索（用户明确说"搜一下"、"就这个了"）
    2. 达到最大澄清轮次 → 强制触发搜索（兜底）
    3. 否则 → 继续澄清
    """
    if not isinstance(subtask, ShoppingSubTask):
        # 非购物意图不需要澄清
        subtask.status = TaskStatus.READY
        return

    # 用户明确表示要立即搜索
    if intent_item.explicit_search:
        subtask.status = TaskStatus.READY
        return

    # 达到最大澄清次数，强制执行
    if subtask.clarification_count >= max_clarification_count:
        subtask.status = TaskStatus.READY
    else:
        # 继续澄清
        subtask.status = TaskStatus.CLARIFYING


async def intent_analyze(llm, user_query: str, subtasks: list[SubTask]) -> IntentResponse:
    """
    意图分解和识别，同时提取槽位信息
    Args:
        - user_query: 根据历史消息重写过后的 query
        - subtasks: 历史的 SubTask 列表
    """
    parser = PydanticOutputParser(pydantic_object=IntentResponse)

    # 构建历史 subtasks 上下文
    history_context = ""
    if subtasks:
        shopping_tasks = [t for t in subtasks if t.category == IntentCategory.SHOPPING]
        if shopping_tasks:
            history_context = "\n\n用户之前讨论过的购物意图如下（用于判断当前意图是否为新意图）：\n"
            for task in shopping_tasks:
                if isinstance(task, ShoppingSubTask):
                    history_context += f"- task_id: {task.task_id}\n"
                    history_context += f"  sub_query: {task.sub_query}\n"
                    history_context += f"  product_category: {task.product_category}\n"
                    history_context += f"  keywords: {task.keywords}\n"
                    history_context += f"  filters: {task.filters}\n"
                    history_context += f"  has_recommended_product_ids: {task.has_recommended_product_ids}\n"
                    history_context += f"  created_at: {task.created_at}\n\n"

    INTENT_ANALYSIS_PROMPT = """
    # Role
    你是一个电商领域的智能意图分析与问题分解专家。你的任务是将用户的输入拆解为独立的原子子问题，并为每个子问题分配准确的意图类别，同时提取购物相关的槽位信息。

    # Intent Categories Definition
    请严格基于以下四个类别进行分类：
    1. **SHOPPING**:
       - 定义：与商品查找、推荐、比价、属性确认（比如购物时，用户澄清自己的需求）的意图。
       - 典型场景："我想买跑鞋"、"iPhone 15 多少钱"、"有没有红色的裙子"、"推荐适合送老人的礼物"。

    2. **PLATFORM**:
       - 定义：与平台政策、交易流程、售后服务、物流、支付、账号操作相关的意图。
       - 典型场景："怎么申请退款"、"发货需要几天"、"运费险怎么赔"、"如何修改收货地址"、"会员有什么权益"、"假货怎么投诉"。

    3. **CHITCHAT**:
       - 定义：非任务型的闲聊、问候、情感交流，或与电商完全无关的通用知识问答，以及泛泛的品牌/品类比较讨论。
       - 典型场景："你好"、"今天天气不错"、"讲个笑话"、"你是谁"、"地球为什么是圆的"。
       - **比较类讨论（走 CHITCHAT）**：
         - 用户说了具体品牌名进行比较，如"华为和iPhone选哪个"、"联想和戴尔哪个好"
         - 用户问的是泛泛的选择问题，如"休闲裤还是工装裤"、"空调和电扇用哪个"
         - 用户只表达对商品的感受，如"这件商品不错"、"那个好看"

    4. **COMPARISON**:
       - 定义：用户使用代词指代我们已推荐的商品，需要进行具体商品比较的场景。
       - **触发条件（必须同时满足）**：
         1. 用户使用了代词（"这几个"、"那些"、"那个"等）
         2. 历史的 ShoppingSubTask 中 has_recommended_product_ids 非空
       - 典型场景："这几个有什么区别"、"那几个呢"、"推荐买哪个"

    # Decomposition Rules (关键步骤)
    1. **原子化拆分**: 如果用户的一句话包含多个独立的需求（例如："我想买双鞋，另外问问怎么退货"），必须将其拆分为两个独立的 `IntentItem`。
       - 错误做法：将整句标记为 SHOPPING。
       - 正确做法：生成两个 item: [{{"sub_query": "我想买双鞋", "intent": "SHOPPING"}}, {{"sub_query": "怎么退货", "intent": "PLATFORM"}}]。
       - 再比如：用户说“给我推荐一只口红和一本小说”，生成两个 item，因为是两个购物请求，[{{"sub_query": "我想买一只口红", "intent": "SHOPPING"}}, {{"sub_query": "我想买一本小说", "intent": "SHOPPING"}}]。

    2. **优先级与过滤**:
       - 如果用户输入纯属闲聊（多个子问题都是闲聊），只返回一个 CHITCHAT 类型的 item。
       - 如果混合了闲聊和业务需求（如 "你好，我想买电脑"），则保留业务需求的 item 以及 闲聊的 item。

    # Intent Matching Rules (SHOPPING 类型的关键判断)
    对于 **SHOPPING** 类型的 intent，需要判断是否是新的购物意图：
    - **新意图** (`is_new=true`): 用户开始了一个全新的购物话题，与历史购物意图完全不同。
    - **旧意图** (`is_new=false`, `matched_task_id=xxx`): 用户在澄清、补充、修改已有的购物意图（按创建时间task_created_at字段，从最近的开始匹配）。
       - 典型情况：用户后续只提供槽位信息（颜色、价格、品牌等），如"红色的"、"100元以内"
       - 典型情况：用户修改需求，如"不要了，我要个便宜的"、"换成蓝色的"
       - 典型情况：用户基于之前推荐的商品追问，如"有没有更便宜的"、"这个尺寸大一点的"
       - 典型情况：用户继续讨论同类别商品，如"再看看其他的"

    # Slot Extraction Rules (仅适用于 SHOPPING 意图)
    对于 SHOPPING 类型的意图，需要从用户输入中提取槽位信息：

    1. **product_category（品类）**: 商品的最小售卖品类，如手机、耳机、电脑、口红等
       - 用户明确说："推荐手机" → product_category: "手机"
       - 用户说"这个有便宜的吗" + 历史有品类 → 沿用历史品类

    2. **keywords（关键词）**: 搜索关键词/扩展词，如拍照好看、续航久、红色、华为等
       - 用户说"拍照好看的" → keywords: ["拍照好看"]
       - 用户说"续航久" → keywords: ["续航久"]

    3. **filters（过滤条件）**: 明确的数值条件
       - "预算3000以内" → price_max: 3000
       - "100元以上" → price_min: 100
       - "要2个" → quantity: 2

    # 重要：对于旧意图，keywords 必须返回完整的列表
    # 即：基于历史的 keywords + 用户当前新增的关键词
    # 不要区分替换还是追加，LLM 根据语义理解自动合并

    # Output Format
    - 必须输出合法的 JSON，符合提供的 Pydantic schema。
    - `intent_items` 列表不能为空。
    - `intent` 字段必须是枚举值之一：SHOPPING, PLATFORM, CHITCHAT, COMPARISON。
    - 对于 PLATFORM 和 CHITCHAT 类型，`is_new` 固定为 true，`matched_task_id` 为 null，`extracted_slots` 为空字典，`explicit_search` 固定为 false，`is_replace_products` 固定为 false。
    - 对于 COMPARISON 类型，`is_new` 固定为 true，`matched_task_id` 为匹配到的 ShoppingSubTask 的 task_id，`extracted_slots` 为空字典。
    - `extracted_slots` 字段必须包含：product_category, keywords, filters（SHOPPING 意图）。
    - `explicit_search` 字段表示用户是否明确表示要立即搜索商品：
      - explicit_search=true：用户说"搜一下"、"款式随意"、"差不多就行"、"直接推荐吧"、"不知道要什么，直接搜索吧"
      - explicit_search=false：用户只是继续描述需求，如"推荐一款"、"要拍照好看的"、"有没有..."
    - `is_replace_products` 字段表示用户是否要求"换一批"：
      - is_replace_products=true：用户明确说"换一批"、"换一个"、"再看看其他的"
      - is_replace_products=false：用户修改条件、补充需求、或首次推荐
    - **重要**：对于 is_new=false 的旧意图，keywords 必须返回完整的列表（历史 + 新增）

    # Few-Shot Examples

    ## 示例1：多意图拆分（新意图，需要澄清）
    User: "推荐几款降噪耳机，顺便问问京东白条怎么开通？"
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "推荐几款降噪耳机",
          "intent": "SHOPPING",
          "is_new": true,
          "matched_task_id": null,
          "extracted_slots": {{
            "product_category": "降噪耳机",
            "keywords": [],
            "filters": {{}}
          }},
          "explicit_search": false,
          "is_replace_products": false
        }},
        {{
          "sub_query": "京东白条怎么开通",
          "intent": "PLATFORM",
          "is_new": true,
          "matched_task_id": null,
          "extracted_slots": {{}},
          "explicit_search": false,
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例2：明确触发搜索
    User: "帮我搜一下降噪耳机，要索尼的"
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "帮我搜一下降噪耳机，要索尼的",
          "intent": "SHOPPING",
          "is_new": true,
          "matched_task_id": null,
          "extracted_slots": {{
            "product_category": "降噪耳机",
            "keywords": ["索尼"],
            "filters": {{}}
          }},
          "explicit_search": true,
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例3：旧意图 - 增加价格条件，但不想搜索
    User: "我的预算是300元钱。"
    Historical Shopping Subtasks:
    - task_id: task_001
      original_query: 推荐几款降噪耳机
      product_category: 降噪耳机
      keywords: []
      filters: {{}}
      clarification_count: 0
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "我的预算是300元钱",
          "intent": "SHOPPING",
          "is_new": false,
          "matched_task_id": "task_001",
          "extracted_slots": {{
            "product_category": "降噪耳机",
            "keywords": [],
            "filters": {{"price_max": 3000}}
          }},
          "explicit_search": false,
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例4：旧意图 - 替换颜色（完整关键词）
    User: "换成红色的。"
    Historical Shopping Subtasks:
    - task_id: task_002
      original_query: 推荐几款口红
      product_category: 口红
      keywords: []
      filters: {{}}
      clarification_count: 0
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "换成红色的",
          "intent": "SHOPPING",
          "is_new": false,
          "matched_task_id": "task_002",
          "extracted_slots": {{
            "product_category": "口红",
            "keywords": ["红色"],
            "filters": {{}}
          }},
          "explicit_search": false,
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例5：旧意图 - 增量添加关键词（完整列表）
    User: "要拍照好看的。"
    Historical Shopping Subtasks:
    - task_id: task_003
      original_query: 推荐几款手机
      product_category: 手机
      keywords: []
      filters: {{}}
      clarification_count: 0
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "要拍照好看的",
          "intent": "SHOPPING",
          "is_new": false,
          "matched_task_id": "task_003",
          "extracted_slots": {{
            "product_category": "手机",
            "keywords": ["拍照好看"],
            "filters": {{}}
          }},
          "explicit_search": false,
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例6：关键示例 - 替换颜色但保留之前的关键词（完整列表）
    User: "换成黑色的，不要红色的。"
    Historical Shopping Subtasks:
    - task_id: task_004
      original_query: 续航久、拍照好看的红色手机
      product_category: 手机
      keywords: ["续航久", "拍照好看", "红色"]
      filters: {{}}
      clarification_count: 0
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "换成黑色的，不要红色的",
          "intent": "SHOPPING",
          "is_new": false,
          "matched_task_id": "task_004",
          "extracted_slots": {{
            "product_category": "手机",
            "keywords": ["续航久", "拍照好看", "黑色"],
            "filters": {{}}
          }},
          "explicit_search": false,
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例7：换一批 - 继续浏览更多商品
    User: "再看看其他的。"
    Historical Shopping Subtasks:
    - task_id: task_001
      original_query: 推荐几款降噪耳机
      product_category: 降噪耳机
      keywords: ["索尼"]
      filters: {{}}
      clarification_count: 0
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "再看看其他的",
          "intent": "SHOPPING",
          "is_new": false,
          "matched_task_id": "task_001",
          "extracted_slots": {{
            "product_category": "降噪耳机",
            "keywords": [],
            "filters": {{}}
          }},
          "explicit_search": false,
          "is_replace_products": true
        }}
      ]
    }}

    ## 示例8：新意图 - 新品类
    User: "我想买个键盘。"
    Historical Shopping Subtasks:
    - task_id: task_001
      original_query: 推荐几款降噪耳机
      product_category: 降噪耳机
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "我想买个键盘",
          "intent": "SHOPPING",
          "is_new": true,
          "matched_task_id": null,
          "extracted_slots": {{
            "product_category": "键盘",
            "keywords": [],
            "filters": {{}}
          }},
          "explicit_search": false,
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例9：闲聊
    User: "你好呀，今天天气不错"
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "你好呀，今天天气不错",
          "intent": "CHITCHAT",
          "is_new": true,
          "matched_task_id": null,
          "extracted_slots": {{}},
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例10：品牌级比较 - 走 CHITCHAT
    User: "华为和iPhone选哪个好？"
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "华为和iPhone选哪个好？",
          "intent": "CHITCHAT",
          "is_new": true,
          "matched_task_id": null,
          "extracted_slots": {{}},
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例11：泛泛的选择讨论 - 走 CHITCHAT
    User: "休闲裤还是工装裤好看？"
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "休闲裤还是工装裤好看？",
          "intent": "CHITCHAT",
          "is_new": true,
          "matched_task_id": null,
          "extracted_slots": {{}},
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例12：商品比较 - 走 COMPARISON（有代词 + 有推荐商品）
    User: "这几个有什么区别？"
    Historical Shopping Subtasks:
    - task_id: task_001
      original_query: 推荐几款电脑
      product_category: 电脑
      keywords: []
      filters: {{}}
      has_recommended_product_ids: [1001, 1002, 1003]
      clarification_count: 0
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "这几个有什么区别？",
          "intent": "COMPARISON",
          "is_new": true,
          "matched_task_id": "task_001",
          "extracted_slots": {{}},
          "is_replace_products": false
        }}
      ]
    }}

    ## 示例13：购物感受 - 走 CHITCHAT
    User: "这件商品不错"
    Assistant:
    {{
      "intent_items": [
        {{
          "sub_query": "这件商品不错",
          "intent": "CHITCHAT",
          "is_new": true,
          "matched_task_id": null,
          "extracted_slots": {{}},
          "is_replace_products": false
        }}
      ]
    }}

    # Historical Shopping Subtasks
    {history_context}


    你的输出格式应严格按照以下要求，不要包含任何额外解释或文本：
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_ANALYSIS_PROMPT),
        ("human", "{user_input}")
    ])
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions(),
        history_context=history_context
    )
    chain = prompt | llm | parser
    return chain.invoke({"user_input": user_query})



def is_new_intent(intent_item: IntentItem, subtasks: list[SubTask]) -> tuple[bool, SubTask | None]:
    """
    判断是否为新意图，并返回匹配到的 SubTask（如果有）
    如果本次 query 是某个 subtask 的延续，说明针对的话题是旧意图，返回旧意图；如果是新意图，则返回 none
    Args:
        intent_item: LLM 解析出的意图项，已包含 is_new 和 matched_task_id
        subtasks: 历史子任务列表

    Returns:
        (是否为新意图, 匹配到的 SubTask 或 None)
    """
    if intent_item.is_new:
        return True, None

    # 如果不是新意图，根据 matched_task_id 查找对应的 SubTask
    if intent_item.matched_task_id:
        for task in subtasks:
            if task.task_id == intent_item.matched_task_id:
                return False, task

    # 如果没有找到匹配的任务，视为新意图（兜底逻辑）
    return True, None
