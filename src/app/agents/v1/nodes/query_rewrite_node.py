"""
@File       : query_rewrite_node.py
@Description: 查询重写节点 - 将用户的问题根据历史消息进行指代消除和信息补全

@Time       : 2026/3/8 23:18
@Author     : hcy18
"""
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from app.agents.v1.schema import (
    ShopmindAgentState,
    ShopmindAssistantContext,
    ShoppingSubTask,
    TaskStatus,
)
from app.agents.v1.utils import build_history_context
from app.utils.logger import app_logger as logger


def _get_all_clarifying_tasks(subtasks: list) -> list[ShoppingSubTask]:
    """获取所有 CLARIFYING 状态的 ShoppingSubTask"""
    return [
        t for t in subtasks
        if isinstance(t, ShoppingSubTask) and t.status == TaskStatus.CLARIFYING
    ]


async def query_rewritten_node(state: ShopmindAgentState, runtime: Runtime[ShopmindAssistantContext]):
    """
    查询重写节点（Agent 第一个节点）
    任务：将用户的问题，根据历史消息，返回信息齐全的 query
    - 指代消除（"这个东西有便宜的吗" -> "便宜的手机"）
    - 信息补全（结合历史对话，补全缺失信息）
    - 多轮澄清场景：感知 CLARIFYING 状态的购物任务，补全商品品类

    注意：多意图拆分由下一个意图分解节点处理
    """
    context = runtime.context
    logger.info(f"[query_rewrite_node] thread_id: {context.thread_id}")

    llm = context.llm
    messages = state.get("messages", [])
    history_text = build_history_context(messages)
    sub_tasks = state.get("sub_tasks", [])
    clarifying_tasks = _get_all_clarifying_tasks(sub_tasks)

    rewritten_query = await query_rewrite(
        llm,
        state.get("original_query", ""),
        history_text,
        clarifying_tasks,
    )

    logger.info(f"[query_rewrite_node] thread_id: {context.thread_id}, rewritten_query: {rewritten_query}")
    return {"rewritten_query": rewritten_query}


def _build_clarifying_context(clarifying_tasks: list[ShoppingSubTask]) -> str:
    """构建多轮澄清场景的上下文"""
    if not clarifying_tasks:
        return ""

    tasks_desc = []
    for i, task in enumerate(clarifying_tasks, 1):
        product_category = task.product_category or "未知品类"
        tasks_desc.append(f"  任务{i}：帮用户选购 [{product_category}]，澄清轮次：第 {task.clarification_count + 1} 轮")

    return f"""
    # 【多轮澄清场景】（当存在等待澄清的购物任务时适用）
    当前用户可能正在回答一个澄清问题，也可能是在问一个新问题。

    【当前澄清中的任务】
    {chr(10).join(tasks_desc)}

    请先判断用户的语义：
    - 如果用户的回复是在**回答**上述某个澄清问题（如对品类、品牌、款式等的具体回答）→ 补全为对应商品品类描述
    - 如果用户是在**切换话题**（闲聊、平台规则、新的独立问题）→ 忽略澄清任务，保持原样重写
    """


async def query_rewrite(
    llm,
    user_query: str,
    history_context: str,
    clarifying_tasks: list[ShoppingSubTask] | None = None,
) -> str:
    """
    使用 LLM 进行查询重写
    输出：重写后的完整 query 字符串

    Args:
        llm: 语言模型
        user_query: 用户原始问题
        history_context: 历史对话上下文
        clarifying_tasks: 所有 CLARIFYING 状态的购物任务（用于多轮澄清场景的信息补全）
    """
    if clarifying_tasks is None:
        clarifying_tasks = []

    clarifying_context = _build_clarifying_context(clarifying_tasks)

    QUERY_REWRITE_PROMPT = f"""
    # Role
    你是一个电商导购助手的查询重写专家。你的任务是务必结合**历史对话**(若有)，将用户的原始问题进行重写，消除指代、补充信息，使查询信息齐全。

    # 核心任务
    1. **指代消除**: 将模糊的指代（如"这个"、"那个"、"东西"、"便宜的"、"换一个"）替换为具体的商品品类或属性
    2. **信息补全**: 结合历史对话，补全用户未明确说明但隐含的信息
    3. **保持原样**: 如果没有问题需要重写（如首次提问），则保持原样；
    4. **多个子问题**: 用户问题如果包含多个子问题，则每个子问题都要分别重写
    5. **保持提问气质**: 如果用户确实在提问，你需要在信息补全后，保持提问的句式和语气，而不是把问句变成陈述句

    # 重写规则
    1. 指代词必须替换为具体的商品品类或之前提到的商品
    2. 如果历史中有相关商品信息，需要补充到当前问题中
    3. 如果是闲聊或首次提问，保持原样即可
    4. **多轮澄清场景**: 用户的回复如果是回答澄清问题，应将回复补全为对应商品品类的描述；如果用户切换话题（闲聊/平台规则），则忽略澄清任务

    {clarifying_context}

    # Few-Shot Examples

    ## 示例1：指代消除
    User: "这个有便宜点的吗？"
    History: 用户: 推荐一款拍照好看的手机，款式随便
            助手: 为您推荐...
    Output: 拍照好看的手机中便宜的选择

    ## 示例2：信息补全
    User: "预算300元以内"
    History: 用户: 推荐几款口红
            助手: 您的预算多少？
    Output: 300元以内的口红

    ## 示例3：信息补全
    User: "随便"
    History: 用户: 推荐几款口红
            助手: 你想要什么用的款式...
    Output: 推荐几款口红, 款式随便

    ## 示例4：指代替换 + 属性补充
    User: "换成蓝色的"
    History: 用户: 推荐iPhone15
            助手: 为您推荐...
    Output: 蓝色的iPhone15

    ## 示例5：继续浏览
    User: "再看看其他的便宜款式"
    History: 用户: 推荐几款降噪耳机
            助手: 为您推荐...
    Output: 其他款式的降噪耳机中便宜的选择

    ## 示例6：闲聊
    User: "你好呀，今天天气不错"
    History: 无历史消息
    Output: 你好呀，今天天气不错

    ## 示例7：首次提问
    User: "推荐一款无线耳机"
    History: 无历史消息
    Output: 推荐一款无线耳机

    ## 示例8：平台问题
    User: "怎么申请退货？"
    History: 无历史消息
    Output: 怎么申请退货？

    ## 示例9：一次性问了多个子问题（每个子问题，分别考虑重写规则）
    User: "我想要稍微便宜的。另外如果不满意，可以支持退货吗？"
    History: 用户: 我想买一部华为手机
             助手: 你的预算是多少
    Output: 我想要一部稍微便宜的华为手机。如果不满意，可以支持退货吗？

    ## 示例10：回应澄清问题（多轮澄清场景）
    User: "我想学习 Python"
    Clarifying Tasks:
      任务1：帮用户选购 [书籍]，澄清轮次：第1轮
    Output: 学习 Python 相关的书籍

    ## 示例11：澄清场景中切换话题（应忽略澄清任务）
    User: "怎么申请退货？"
    Clarifying Tasks:
      任务1：帮用户选购 [书籍]，澄清轮次：第1轮
    Output: 怎么申请退货？

    # 当前对话
    ## 历史对话
    {history_context}


    # Output 要求
    - 直接输出重写后的查询文本，不要包含任何解释、JSON 或其他格式
    - 如果无需重写，直接输出原始问题
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITE_PROMPT),
        ("human", "用户问题：{user_input}")
    ])
    prompt = prompt.partial(
        history_context=history_context,
    )
    chain = prompt | llm
    result = await chain.ainvoke({"user_input": user_query})
    return result.content.strip()
