"""
@File       : query_rewrite_node.py
@Description: 查询重写节点 - 将用户的问题根据历史消息进行指代消除和信息补全

@Time       : 2026/3/8 23:18
@Author     : hcy18
"""
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from app.agents.v3.schema import (
    ShopmindAgentState,
    ShopmindAssistantContext,
)
from app.utils.logger import app_logger as logger


async def query_rewritten_node(state: ShopmindAgentState, runtime: Runtime[ShopmindAssistantContext]):
    """
    查询重写节点（Agent 第一个节点）
    任务：将用户的问题，根据历史消息，返回信息齐全的 query
    - 指代消除（"这个东西有便宜的吗" -> "便宜的手机"）
    - 信息补全（结合历史对话，补全缺失信息）

    注意：多意图拆分由下一个意图分解节点处理
    """
    context = runtime.context
    llm = context.llm

    # 获取历史消息（用于指代消除和信息补全）
    messages = state.messages
    history_text = _build_history_context(messages)

    # 调用 LLM 进行查询重写
    rewritten_query = await query_rewrite(llm, state.original_query, history_text)

    # 更新 state
    state.rewritten_query= rewritten_query

    logger.info(f"thread_id: {context.thread_id}, 查询重写结果: {rewritten_query}")
    return state


def _build_history_context(messages: list) -> str:
    """构建历史对话上下文"""
    if not messages:
        return "无历史消息"

    history_parts = []
    for msg in messages[-5:]:  # 只取最近5条消息
        role = "用户" if msg.type == "human" else "助手"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # 截断过长内容
        if len(content) > 200:
            content = content[:200] + "..."
        history_parts.append(f"{role}: {content}")

    return "\n".join(history_parts)


async def query_rewrite(
    llm,
    user_query: str,
    history_context: str,
) -> str:
    """
    使用 LLM 进行查询重写
    输出：重写后的完整 query 字符串
    """
    QUERY_REWRITE_PROMPT = """
    # Role
    你是一个电商导购助手的查询重写专家。你的任务是将用户的原始问题进行重写，消除指代、补充信息，使查询信息齐全。

    # 核心任务
    1. **指代消除**: 将模糊的指代（如"这个"、"那个"、"东西"、"便宜的"、"换一个"）替换为具体的商品品类或属性
    2. **信息补全**: 结合历史对话，补全用户未明确说明但隐含的商品信息
    3. **保持原样**: 如果没有问题需要重写（如首次提问），则保持原样
    4. **多个子问题**: 用户问题如果包含多个子问题，则每个子问题都要分别重写

    # 重写规则
    1. 指代词必须替换为具体的商品品类或之前提到的商品
    2. 如果历史中有相关商品信息，需要补充到当前问题中
    3. 如果是闲聊或首次提问，保持原样即可

    # Few-Shot Examples

    ## 示例1：指代消除
    User: "这个有便宜点的吗？"
    History: 用户: 推荐一款拍照好看的手机
            助手: 为您推荐...
    Output: 拍照好看的手机中便宜的选择

    ## 示例2：信息补全
    User: "预算300元以内的"
    History: 用户: 推荐几款口红
            助手: 为您推荐...
    Output: 300元以内的口红

    ## 示例3：指代替换 + 属性补充
    User: "换成蓝色的"
    History: 用户: 推荐iPhone15
            助手: 为您推荐...
    Output: 蓝色的iPhone15

    ## 示例4：继续浏览
    User: "再看看其他的便宜款式"
    History: 用户: 推荐几款降噪耳机
            助手: 为您推荐...
    Output: 其他款式的降噪耳机中便宜的选择

    ## 示例5：闲聊
    User: "你好呀，今天天气不错"
    History: 无历史消息
    Output: 你好呀，今天天气不错

    ## 示例6：首次提问
    User: "推荐一款无线耳机"
    History: 无历史消息
    Output: 推荐一款无线耳机

    ## 示例7：平台问题
    User: "怎么申请退货？"
    History: 无历史消息
    Output: 怎么申请退货？
    
    ## 示例8：一次性问了多个子问题（每个子问题，分别考虑重写规则）
    User: "我想要稍微便宜的。另外如果不满意，可以支持退货吗？"
    History: 用户: 我想买一部华为手机
             助手: 你的预算是多少
    Output: 我想要一部稍微便宜的华为手机。如果不满意，可以支持退货吗？

    # 当前对话
    ## 历史对话
    {history_context}

    ## 用户当前问题
    {user_query}

    # Output 要求
    - 直接输出重写后的查询文本，不要包含任何解释、JSON 或其他格式
    - 如果无需重写，直接输出原始问题
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITE_PROMPT),
        ("human", "{user_input}")
    ])
    prompt = prompt.partial(
        history_context=history_context,
    )
    chain = prompt | llm
    result = await chain.ainvoke({"user_input": user_query})
    return result.content.strip()
