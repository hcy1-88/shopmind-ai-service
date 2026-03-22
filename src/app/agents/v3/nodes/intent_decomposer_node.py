"""
@File       : intent_decomposer_node.py
@Description:

@Time       : 2026/3/8 21:42
@Author     : hcy18
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from app.agents.v3.schema import ShopmindAssistantContext, ShopmindAgentState, IntentResponse, IntentItem, SubTask, IntentCategory, TaskStatus
from app.utils.logger import app_logger as logger


async def intent_decomposer_node(state: ShopmindAgentState, runtime: Runtime[ShopmindAssistantContext]):
    """意图分解器节点，负责把用户的 query 分解成 subtask 列表"""
    context = runtime.context
    llm = context.llm
    # 1, 意图识别：购物、平台规则、闲聊
    subtasks = state.get("sub_tasks", [])
    intent_resp = await intent_analyze(llm, state["rewritten_query"], subtasks)
    logger.info(f"thread_id: {context.thread_id}, 意图识别结果：{intent_resp}")
    for intent_item in intent_resp.intent_items:
        # 2，如果是购物，则需要判断是新的购物意图，还是旧的购物意图
        ## 2.1 如果是新的购物意图，则新增 subtask
        ## 2.2 如果是旧的购物意图，则针对相应的 subtask ，填充槽位，状态修改为 READY，重新搜索
        is_new, matched_subtask = is_new_intent(intent_item, subtasks)

        if intent_item.intent == IntentCategory.SHOPPING:
            if is_new:
                # 创建新的 SubTask
                new_subtask = SubTask(
                    category=intent_item.intent,
                    original_query=intent_item.sub_query,
                    status=TaskStatus.NEW,
                )
                state["sub_tasks"].append(new_subtask)
            else:
                # 更新已有的 SubTask
                matched_subtask.status = TaskStatus.READY

        # 3，平台规则，则交给 RAG Agent Node

        # 4，闲聊，则交给 Chitchat Agent Node


async def intent_analyze(llm, user_query: str, subtasks: list[SubTask]) -> IntentResponse:
    """意图分解和识别"""
    parser = JsonOutputParser(pydantic_object=IntentResponse)

    # 构建历史 subtasks 上下文
    history_context = ""
    if subtasks:
        shopping_tasks = [t for t in subtasks if t.category == IntentCategory.SHOPPING]
        if shopping_tasks:
            history_context = "\n\n用户之前讨论过的购物意图如下（用于判断当前意图是否为新意图）：\n"
            for task in shopping_tasks:
                history_context += f"- task_id: {task.task_id}\n  original_query: {task.original_query}\n  filled_slots: {task.filled_slots}\n task_created_at: {task.created_at }\n\n"

    INTENT_ANALYSIS_PROMPT = """
    # Role
    你是一个电商领域的智能意图分析与问题分解专家。你的任务是将用户的输入拆解为独立的原子子问题，并为每个子问题分配准确的意图类别。

    # Intent Categories Definition
    请严格基于以下三个类别进行分类：
    1. **SHOPPING**:
       - 定义：与商品查找、推荐、比价、属性确认（比如购物时，用户澄清自己的需求）的意图。
       - 典型场景："我想买跑鞋"、"iPhone 15 多少钱"、"有没有红色的裙子"、"推荐适合送老人的礼物"。

    2. **PLATFORM**:
       - 定义：与平台政策、交易流程、售后服务、物流、支付、账号操作相关的意图。
       - 典型场景："怎么申请退款"、"发货需要几天"、"运费险怎么赔"、"如何修改收货地址"、"会员有什么权益"、"假货怎么投诉"。

    3. **CHITCHAT**:
       - 定义：非任务型的闲聊、问候、情感交流，或与电商完全无关的通用知识问答。
       - 典型场景："你好"、"今天天气不错"、"讲个笑话"、"你是谁"、"地球为什么是圆的"。

    # Decomposition Rules (关键步骤)
    1. **原子化拆分**: 如果用户的一句话包含多个独立的需求（例如："我想买双鞋，另外问问怎么退货"），必须将其拆分为两个独立的 `IntentItem`。
       - 错误做法：将整句标记为 SHOPPING。
       - 正确做法：生成两个 item: [{"sub_query": "我想买双鞋", "intent": "SHOPPING"}, {"sub_query": "怎么退货", "intent": "PLATFORM"}]。

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

    # Output Format
    - 必须输出合法的 JSON，符合提供的 Pydantic  schema。
    - `intent_items` 列表不能为空。
    - `intent` 字段必须是枚举值之一：SHOPPING, PLATFORM, CHITCHAT。
    - 对于 PLATFORM 和 CHITCHAT 类型，`is_new` 固定为 true，`matched_task_id` 为 null。

    # Few-Shot Examples

    User: "推荐几款降噪耳机，顺便问问京东白条怎么开通？"
    Assistant:
    {
      "intent_items": [
        {
          "sub_query": "推荐几款降噪耳机",
          "intent": "SHOPPING",
          "is_new": true,
          "matched_task_id": null
        },
        {
          "sub_query": "京东白条怎么开通",
          "intent": "PLATFORM",
          "is_new": true,
          "matched_task_id": null
        }
      ]
    }

    User: "我上周买的衣服尺码小了，想换货，流程是怎样的？另外有没有新款的 T 恤推荐？"
    Assistant:
    {
      "intent_items": [
        {
          "sub_query": "我上周买的衣服尺码小了，想换货，流程是怎样的",
          "intent": "PLATFORM",
          "is_new": true,
          "matched_task_id": null
        },
        {
          "sub_query": "有没有新款的 T 恤推荐",
          "intent": "SHOPPING",
          "is_new": true,
          "matched_task_id": null
        }
      ]
    }

    User: "我的预算是300元钱。"
    Historical Shopping Subtasks:
    - task_id: task_001
      original_query: 推荐几款降噪耳机
      filled_slots: {"keywords": "降噪耳机"}
    Assistant:
    {
      "intent_items": [
        {
          "sub_query": "我的预算是300元钱",
          "intent": "SHOPPING",
          "is_new": false,
          "matched_task_id": "task_001"
        }
      ]
    }

    User: "我要红色的。"
    Historical Shopping Subtasks:
    - task_id: task_002
      original_query: 推荐几款口红
      filled_slots: {"keywords": "口红"}
    Assistant:
    {
      "intent_items": [
        {
          "sub_query": "我要红色的",
          "intent": "SHOPPING",
          "is_new": false,
          "matched_task_id": "task_002"
        }
      ]
    }

    User: "换成蓝牙耳机吧。"
    Historical Shopping Subtasks:
    - task_id: task_001
      original_query: 推荐几款降噪耳机
      filled_slots: {"keywords": "降噪耳机"}
    Assistant:
    {
      "intent_items": [
        {
          "sub_query": "换成蓝牙耳机吧",
          "intent": "SHOPPING",
          "is_new": false,
          "matched_task_id": "task_001"
        }
      ]
    }

    User: "再看看其他的。"
    Historical Shopping Subtasks:
    - task_id: task_001
      original_query: 推荐几款降噪耳机
      filled_slots: {"keywords": "降噪耳机"}
    Assistant:
    {
      "intent_items": [
        {
          "sub_query": "再看看其他的",
          "intent": "SHOPPING",
          "is_new": false,
          "matched_task_id": "task_001"
        }
      ]
    }

    User: "我想买个键盘。"
    Historical Shopping Subtasks:
    - task_id: task_001
      original_query: 推荐几款降噪耳机
      filled_slots: {"keywords": "降噪耳机"}
    Assistant:
    {
      "intent_items": [
        {
          "sub_query": "我想买个键盘",
          "intent": "SHOPPING",
          "is_new": true,
          "matched_task_id": null
        }
      ]
    }

    # Historical Shopping Subtasks
    {history_context}
    
    你的输出格式应严格按照以下要求,不要包含任何额外解释或文本：
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_ANALYSIS_PROMPT),
        ("human", "{user_input}")
    ])
    prompt.partial(format_instructions=parser.get_format_instructions(), history_context=history_context)
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

