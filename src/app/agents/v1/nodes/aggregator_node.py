"""
@File       : aggregator_node.py
@Description: 聚合节点 - 将多个子任务的 final_response 整合成连贯的最终答案

@Time       : 2026/3/23 17:10
@Author     : hcy18
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from app.agents.v1.schema import ShopmindAgentState, ShopmindAssistantContext
from app.utils.logger import app_logger as logger


async def aggregate_node(state: ShopmindAgentState, runtime: Runtime[ShopmindAssistantContext]):
    """聚合多个子任务的响应，生成连贯的最终答案

    示例场景：
    用户说 "请推荐一支口红，款式价格随意。你们支持退货吗？康德的人性哲学观是什么？"
    会触发 3 个 task 并行执行：
    - task 1：shopping 购物，final_response="我为您推荐的口红是..."
    - task 2：平台规则，final_response="我们支持退货..."
    - task 3：闲聊，final_response="康德的人性哲学...."
    此聚合器节点，负责把三者的 final_response 整合成一个连贯的答案，返回给用户。
    """
    context = runtime.context
    llm = context.llm
    thread_id = context.thread_id
    logger.info(f"[AggregateNode] thread_id: {thread_id}")

    # 获取所有子任务的 final_response
    sub_task_results = state.get("sub_task_results", [])

    # 收集有 final_response 的任务响应
    task_responses = []
    for task in sub_task_results:
        if task.final_response:
            task_responses.append({
                "category": task.category.value if hasattr(task.category, "value") else str(task.category),
                "sub_query": task.sub_query,
                "response": task.final_response,
            })

    if not task_responses:
        return {"answer": "抱歉，暂时无法处理您的请求。", "streaming_started": False}

    # 构建聚合 prompt
    if len(task_responses) == 1:
        # 只有一个响应，直接返回，无 LLM 调用，不走流式
        answer = task_responses[0]["response"]
        logger.info(f"[AggregateNode] thread_id: {thread_id}, answer length: {len(answer)}")
        return {"answer": answer, "is_replace_products": False, "streaming_started": False}
    else:
        # 多个响应，调用 LLM 整合，流式输出
        answer = await _generate_coherent_answer(llm, task_responses)

    logger.info(f"[AggregateNode] thread_id: {thread_id}, answer length: {len(answer)}")
    return {"answer": answer, "streaming_started": True}


async def _generate_coherent_answer(llm, task_responses: list[dict]) -> str:
    """调用 LLM 生成连贯的最终答案

    Args:
        llm: 大模型实例
        task_responses: 子任务响应列表，每项包含 category, sub_query, response
    """
    # 构建各任务的响应详情
    responses_detail = []
    for i, tr in enumerate(task_responses, 1):
        responses_detail.append(
            f"【任务 {i}】（{tr['category']}）\n"
            f"子问题：{tr['sub_query']}\n"
            f"回答：{tr['response']}"
        )
    responses_text = "\n\n".join(responses_detail)

    system_prompt = """你是一个电商平台的智能导购助手，需要将多个子任务的回答整合成一个连贯自然的回复。

## 整合规则
1. **保持完整性**：所有子任务的回答都必须保留，不能遗漏任何信息
2. **商品链接格式**：如果回答中包含商品超链接（如 [商品名称](product:12345)），必须原样保留，禁止修改或删除
3. **语言连贯**：将多个回答整合成一段流畅的文字，像真人导购一样自然地回复
4. **结构合理**：如果回答涉及多个独立话题，可以用自然的过渡语连接
5. **不重复**：避免重复表达相同内容

## 回答风格
- 语气友好、专业，像真人导购与顾客交流
- 简洁明了，突出重点
- 如果是购物推荐，强调商品亮点；如果是平台规则，给出清晰指引；如果是闲聊，轻松自然

直接返回整合后的回答，不要添加任何解释说明。"""

    user_prompt = f"""以下是各子任务的回答：

{responses_text}

请将以上回答整合成一个连贯的最终回复。"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    return response.content.strip()