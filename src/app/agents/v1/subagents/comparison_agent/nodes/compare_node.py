"""
@File       : compare_node.py
@Description: 商品比较节点 - 生成对比文案和购买建议

@Time       : 2026/3/27
@Author     : hcy18
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from app.agents.v1.schema import ComparisonSubTask, TaskStatus, ShopmindAssistantContext
from app.utils.logger import app_logger as logger


COMPARISON_SYSTEM_PROMPT = """你是一个专业的电商导购助手，正在为用户对比商品。

## 你的任务
根据商品详情信息，生成清晰的对比文案和购买建议。

## 对比维度
请从以下维度进行对比（根据实际商品信息选择适用的维度）：
1. 价格
2. 品牌
3. 核心参数/规格
4. 适用场景
5. 用户评价/口碑
6. 售后保障

## 输出格式
请按以下格式输出：

## 商品对比
| 维度 | 商品1名称 | 商品2名称 | ... |
|------|----------|----------|-----|
| 价格 | xxx | xxx | ... |
| ... | ... | ... | ... |

## 购买建议
基于用户需求，给出明确的购买建议。如果用户没有明确需求，可以从性价比、适用场景等角度给出建议。

## 注意事项
- 只对比商品详情中确实存在的信息，不要编造
- 如果某个维度信息不全，在对比表中标注"未提供"
- 购买建议要具体、有针对性，不要泛泛而谈
"""


async def compare_node(state: dict, runtime: Runtime[ShopmindAssistantContext]) -> dict:
    """
    商品比较节点 - 生成对比文案和购买建议

    输入:
        state: ComparisonSubgraphState
            - task: ComparisonSubTask
            - product_details: list[ProductResponseDto] 商品详情列表

    输出:
        dict: 更新 task.final_response
    """
    task: ComparisonSubTask = state.get("task")
    product_details = state.get("product_details", [])

    logger.info(f"[compare_node] task_id: {task.task_id}")

    if not product_details:
        task.final_response = "没有找到要比较的商品信息"
        task.status = TaskStatus.COMPLETED
        return {"task": task}

    # 构建商品详情文本
    product_details_list = []
    for dto in product_details:
        product_details_list.append(dto.model_dump_json(indent=2))
    product_details_text = "\n\n".join(product_details_list)

    # 构建 prompt
    user_prompt = f"""## 待比较商品详情
{product_details_text}

请根据以上商品详情，生成对比文案和购买建议。"""

    # 调用 LLM 生成对比文案
    llm = runtime.context.llm
    response = await llm.ainvoke([
        SystemMessage(content=COMPARISON_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    # 设置 final_response
    task.final_response = response.content.strip()

    return {"task": task}
