"""
@File       : intent_decomposer_node.py
@Description: 意图分解节点 - 负责把用户的 query 分解成 subtask 列表，并提取槽位信息

@Time       : 2026/3/8 21:42
@Author     : hcy18

四步线性流水线设计：
Step 1: 意图分类 + 原子拆分（基于 rewritten_query）
Step 2: 历史任务匹配（4 种意图都参与）
Step 3: 槽位提取（仅 SHOPPING）
Step 4: 搜索意愿判断（仅 SHOPPING）
"""
from typing import Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_config
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
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
from app.agents.v1.config import (
    MAX_CLARIFICATION_COUNT,
    MAX_HISTORY_TASK_COUNT,
)
from app.utils.logger import app_logger as logger


# ======================= 中间步骤数据结构 =======================

class Step1Item(BaseModel):
    """Step 1 输出：意图拆分 + 分类"""
    sub_query: str = Field(description="原子化拆分后的子查询")
    intent: IntentCategory = Field(description="意图类型：SHOPPING / PLATFORM / CHITCHAT / COMPARISON")


class Step2Item(BaseModel):
    """Step 2 输出：历史任务匹配"""
    sub_query: str
    intent: IntentCategory
    is_new: bool = Field(description="是否为新任务")
    matched_task_id: Optional[str] = Field(default=None, description="匹配到的历史任务 ID")


class Step3Item(BaseModel):
    """Step 3 输出：槽位提取（仅 SHOPPING）"""
    sub_query: str
    intent: IntentCategory
    is_new: bool
    matched_task_id: Optional[str] = None
    product_category: Optional[str] = Field(default=None, description="商品品类")
    keywords: list[str] = Field(default_factory=list, description="搜索关键词")
    filters: dict = Field(default_factory=dict, description="过滤条件")


class Step4Item(BaseModel):
    """Step 4 输出：搜索意愿判断（仅 SHOPPING）"""
    sub_query: str
    intent: IntentCategory
    is_new: bool
    matched_task_id: Optional[str] = None
    product_category: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    filters: dict = Field(default_factory=dict)
    explicit_search: bool = Field(default=False, description="是否明确要求立即搜索")
    is_replace_products: bool = Field(default=False, description="是否要求换一批")


# ======================= 活跃任务过滤 =======================

ACTIVE_TASK_STATUSES = {TaskStatus.NEW, TaskStatus.CLARIFYING, TaskStatus.READY, TaskStatus.WAITING}


def _filter_active_subtasks(subtasks: list[SubTask], max_count: int) -> list[SubTask]:
    """过滤并限制活跃的历史子任务

    1. 过滤掉已完成的任务（COMPLETED/FAILED）
    2. 限制只发送最近 max_count 个任务
    """
    active_tasks = [t for t in subtasks if t.status in ACTIVE_TASK_STATUSES]
    return active_tasks[-max_count:]


# ======================= Step 1: 意图分类 + 原子拆分 =======================

STEP1_PROMPT = """
# Role
你是一个电商领域的意图分类专家（以消费者视角猜测意图）。你的任务是将用户的查询拆解为独立的原子子问题，并判断每个子问题的意图类型。

# 历史任务上下文
{history_context}

# Intent Categories
1. **SHOPPING**: 商品查找、推荐、比价、属性确认（如"我想买手机"、"有没有红色的裙子"），总之是用户有购物意图，需要搜索商品的一类问题
2. **PLATFORM**: 平台政策、退货物流、支付、账号操作（如"怎么退货"、"会员权益"）
3. **CHITCHAT**: 闲聊、问候、情感交流，或与电商无关的知识问答（如"你好"、"天气不错"）
4. **COMPARISON**: 比较已推荐商品的场景。**判断规则**：当历史任务中 has_recommended_product_ids 非空时，如果 query 是在比较/推荐这些已推荐商品（如"海飞丝和飘柔洗发水有什么区别"），则为 COMPARISON；has_recommended_product_ids 为空时，泛泛的品牌/商品比较属于闲聊（如"华为和苹果哪个好"）。

# 拆解思路（按顺序执行）

**Step 1: 先判断意图数量**
仔细阅读用户问题，判断它在问**几件事**。
- 如果多个子句在表达**同一类事情**（如比较+推荐），算1个意图
- 如果明确问的是**不同类事情**（如购物+退货、同时买多个明确的物品单元），才算多个意图

**Step 2: 再判断每个意图的类型**
确定了数量后，再分别判断每个意图的类型。

**Step 3: 生成 sub_query**
- 1个意图 → 整个 query 或合并后的完整描述作为 sub_query
- 多个意图 → 每个意图一个 sub_query

**Step 4: 回查是否需要合并**
如果发现两个 sub_query 其实是**同一个意图的不同表达**，合并它们。

# Rules
- **宁少勿多**：拆分意图时尽量保守，除非明显是多意图，否则不拆分
- 每个原子子问题只能属于一种意图
- 如果整句是闲聊，返回一个 CHITCHAT 类型的 item
- 如果混合闲聊和业务需求，保留业务需求的 item 和闲聊的 item
- **不要拆分主需求和品类修饰词**："学习Python的书" 不要拆成 "学习Python" + "书"，应该是一个 item："学习Python的书"；同理 "红色连衣裙" 也不要拆成 "红色" + "连衣裙"
- **COMPARISON vs CHITCHAT**：历史任务中 has_recommended_product_ids 非空 + query 比较这些商品 → COMPARISON；否则 → CHITCHAT

# Output Format
输出标准 JSON 对象，包含 items 字段：

```json
{{"items": [{{"sub_query": "...", "intent": "SHOPPING"}}]}}
```

{format_instructions}

# 拆解示例

## 示例1：历史中有已推荐商品 - COMPARISON
History: task_id=task_001, product_category=洗发水, has_recommended_product_ids=[P001,P002]
User: "海飞丝洗发水和飘柔洗发水有什么区别？"
分析：has_recommended_product_ids 非空，且 query 比较的是这些已推荐商品
→ intent=COMPARISON
```json
{{"items": [{{"sub_query": "海飞丝洗发水和飘柔洗发水有什么区别？", "intent": "COMPARISON"}}]}}
```

## 示例2：历史中无已推荐商品 - CHITCHAT
History: 无历史任务
User: "华为和苹果哪个好？"
分析：has_recommended_product_ids 为空，属于泛泛的品牌比较
→ intent=CHITCHAT
```json
{{"items": [{{"sub_query": "华为和苹果哪个好？", "intent": "CHITCHAT"}}]}}
```

## 示例3：真正多意图（需要拆分）
User: "推荐一款洗发水，另外退货流程是什么？"
分析：2件不同类事情
```json
{{"items": [{{"sub_query": "推荐一款洗发水", "intent": "SHOPPING"}}, {{"sub_query": "退货流程是什么？", "intent": "PLATFORM"}}]}}
```

## 示例4：避免过度拆分
User: "学习Python的书"
分析：1件事
```json
{{"items": [{{"sub_query": "学习Python的书", "intent": "SHOPPING"}}]}}
```

## 示例5：闲聊
User: "你好呀，今天天气不错"
```json
{{"items": [{{"sub_query": "你好呀，今天天气不错", "intent": "CHITCHAT"}}]}}
```

## 错误示例（必须避免）

### 错误1: 凭空捏造
User: "学习Python的书"
```json
{{"items": [{{"sub_query": "学习Python的书", "intent": "SHOPPING"}}, {{"sub_query": "学习Python的视频课程", "intent": "SHOPPING"}}]}}
```
❌ 错误：不要凭空捏造用户没有提到的内容！

### 错误2: 过度拆分
User: "学习Python的书，你有什么推荐吗"
```json
{{"items": [{{"sub_query": "学习Python的书", "intent": "SHOPPING"}}, {{"sub_query": "你有什么推荐吗", "intent": "SHOPPING"}}]}}
```
❌ 错误：明明是一个意图，却拆成了两个

### 错误3: 有已推荐商品却判为 CHITCHAT
History: task_id=task_001, has_recommended_product_ids=[P001,P002]
User: "P001和P002哪个好？"
错误判断:
```json
{{"items": [{{"sub_query": "P001和P002哪个好？", "intent": "CHITCHAT"}}]}}
```
❌ 错误：has_recommended_product_ids 非空，query 比较的是已推荐商品，应该是 COMPARISON

"""


async def step1_intent_classify(llm, rewritten_query: str, subtasks: list[SubTask]) -> list[Step1Item]:
    """Step 1: 意图分类 + 原子拆分"""
    from pydantic import create_model

    DynamicStep1Item = create_model(
        "DynamicStep1Item",
        sub_query=(str, ...),
        intent=(IntentCategory, ...),
    )

    class Step1Result(BaseModel):
        items: list[DynamicStep1Item] = Field(description="意图拆分结果列表")

    parser = PydanticOutputParser(pydantic_object=Step1Result)
    history_context = _build_history_context_for_step2(subtasks)
    prompt = ChatPromptTemplate.from_messages([
        ("system", STEP1_PROMPT),
        ("human", "User Query: {query}\n\n{history_context}")
    ])
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions(),
        history_context=history_context,
    )
    chain = prompt | llm | parser

    result = await chain.ainvoke({"query": rewritten_query})
    return [
        Step1Item(sub_query=r.sub_query, intent=r.intent)
        for r in result.items
    ]


# ======================= Step 2: 历史任务匹配 =======================

STEP2_HISTORY_PROMPT = """
# Role
你是一个任务匹配专家。根据用户当前的子查询和历史购物任务，判断是否为新任务。

# Matching Rules
- **SHOPPING**: 匹配同类 ShoppingSubTask（品类相似）；如果是全新品类则为新任务
- **COMPARISON**: 必须匹配 has_recommended_product_ids 非空的历史 ShoppingSubTask
- **PLATFORM**: 匹配关联商品的 ShoppingSubTask（为扩展预留）
- **CHITCHAT**: 匹配关联商品的 ShoppingSubTask（为扩展预留）

# 判断 is_new 的关键
- 用户明确在补充/修改/延续已有需求 → is_new=false
- 用户开启全新话题/品类 → is_new=true
- 注意："我说了随便，请立即搜索推荐"这类模糊回复应匹配到最近的相关任务，is_new=false

# Output Format
输出标准 JSON 对象，wrap 在 ```json ``` 代码块中。示例：

```json
{{"sub_query": "我想买个键盘", "intent": "SHOPPING", "is_new": true, "matched_task_id": null}}
```

# Few-Shot Examples

## 示例1: 新品类
User: "我想买个键盘"
History: task_id=task_001, product_category=耳机
```json
{{"sub_query": "我想买个键盘", "intent": "SHOPPING", "is_new": true, "matched_task_id": null}}
```

## 示例2: 延续旧任务
User: "我说了随便"
History: task_id=task_001, product_category=洗发水, sub_query=推荐一款洗发水
```json
{{"sub_query": "我说了随便", "intent": "SHOPPING", "is_new": false, "matched_task_id": "task_001"}}
```

## 示例3: 商品比较
User: "这几个有什么区别"
History: task_id=task_001, product_category=手机, has_recommended_product_ids=[1001,1002]
```json
{{"sub_query": "这几个有什么区别", "intent": "COMPARISON", "is_new": true, "matched_task_id": "task_001"}}
```
"""


def _build_history_context_for_step2(subtasks: list[SubTask]) -> str:
    """构建 Step2 所需的历史任务上下文"""
    if not subtasks:
        return "无历史任务"

    shopping_tasks = [t for t in subtasks if t.category == IntentCategory.SHOPPING]
    if not shopping_tasks:
        return "无历史任务"

    context = "用户的历史购物任务：\n"
    for task in shopping_tasks:
        if isinstance(task, ShoppingSubTask):
            context += f"- task_id: {task.task_id}\n"
            context += f"  sub_query: {task.sub_query}\n"
            context += f"  product_category: {task.product_category}\n"
            context += f"  keywords: {task.keywords}\n"
            context += f"  has_recommended_product_ids: {task.has_recommended_product_ids}\n"
            context += f"  created_at: {task.created_at}\n\n"
    return context


async def step2_task_match(llm, step1_items: list[Step1Item], subtasks: list[SubTask]) -> list[Step2Item]:
    """Step 2: 历史任务匹配"""
    from pydantic import create_model
    DynamicStep2Item = create_model(
        "DynamicStep2Item",
        sub_query=(str, ...),
        intent=(IntentCategory, ...),
        is_new=(bool, ...),
        matched_task_id=(Optional[str], None),
    )
    parser = PydanticOutputParser(pydantic_object=DynamicStep2Item)
    history_context = _build_history_context_for_step2(subtasks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", STEP2_HISTORY_PROMPT),
        ("human", "User Query: {query}\n\n{history_context}")
    ])

    chain = prompt | llm | parser

    results = []
    for item in step1_items:
        result = await chain.ainvoke({"query": item.sub_query, "history_context": history_context})
        results.append(Step2Item(
            sub_query=result.sub_query,
            intent=result.intent,
            is_new=result.is_new,
            matched_task_id=result.matched_task_id,
        ))
    return results


# ======================= Step 3: 槽位提取（仅 SHOPPING） =======================

STEP3_PROMPT = """
# Role
你是一个槽位提取专家。从 SHOPPING 意图的子查询中提取商品品类、关键词和过滤条件。

# Slot Definitions
- **product_category**: 商品最小品类（如手机、耳机、口红）
- **keywords**: 搜索关键词/扩展词（如拍照好看、续航久、华为）
- **filters**: 明确的数值条件（如 price_max=3000, price_min=100）

# Rules
- 从 sub_query 直接提取槽位信息
- keywords 只提取当前 query 中明确提到的
- filters 只提取明确的数值条件

# 历史槽位合并规则（当提供了历史槽位信息时适用）
当用户继续之前的购物意图时（如"换成蓝色的"、"还要续航久的"），需要基于历史槽位做**增量合并**：
- **替换场景**：如果当前 query 表达了替换意图（如"换成X"、"不要X"、"换个Y"），则从历史 keywords 中移除被替换的词
- **追加场景**：如果当前 query 表达了新增意图（如"还要X"、"加个X"），则追加到历史 keywords
- **filter 替换**：如果当前 query 给出了新的数值条件（如"预算2000以内"），则替换历史的 filter 值
- **无法判断时**：按追加处理，保留历史 keywords

# Output Format
输出标准 JSON 对象，wrap 在 ```json ``` 代码块中。示例：

```json
{{"product_category": "手机", "keywords": ["拍照好看", "华为"], "filters": {{"price_max": 3000}}}}
```

# Few-Shot Examples

## 无历史上下文（新品类）

User: "推荐一款拍照好看的华为手机，预算3000以内"
```json
{{"product_category": "手机", "keywords": ["拍照好看", "华为"], "filters": {{"price_max": 3000}}}}
```

## 有历史上下文（continuation query）

User: "换成蓝色的"
历史 keywords: ["拍照好看", "续航久", "红色"]
```json
{{"product_category": "手机", "keywords": ["拍照好看", "续航久", "蓝色"], "filters": {{}}}}
```

User: "还要续航久的"
历史 keywords: ["拍照好看"]
```json
{{"product_category": "手机", "keywords": ["拍照好看", "续航久"], "filters": {{}}}}
```

User: "预算2000以内"
历史 filters: {{"price_max": 3000}}
```json
{{"product_category": "手机", "keywords": [], "filters": {{"price_max": 2000}}}}
```
"""


async def step3_extract_slots(llm, step2_items: list[Step2Item], subtasks: list[SubTask]) -> list[Step3Item]:
    """Step 3: 槽位提取（仅 SHOPPING）"""
    from pydantic import create_model

    DynamicStep3Item = create_model(
        "DynamicStep3Item",
        product_category=(Optional[str], None),
        keywords=(list[str], None),
        filters=(dict, None),
    )
    parser = PydanticOutputParser(pydantic_object=DynamicStep3Item)

    # 构建 matched_task_id → 历史槽位信息 的映射
    matched_slots: dict[str, dict] = {}
    for t in subtasks:
        if isinstance(t, ShoppingSubTask) and t.task_id:
            matched_slots[t.task_id] = {
                "product_category": t.product_category,
                "keywords": t.keywords or [],
                "filters": t.filters or {},
            }

    # 两个版本的 prompt 模板：有历史 vs 无历史
    prompt_no_history = ChatPromptTemplate.from_messages([
        ("system", STEP3_PROMPT),
        ("human", "User: {query}")
    ])
    prompt_no_history = prompt_no_history.partial(format_instructions=parser.get_format_instructions())
    chain_no_history = prompt_no_history | llm | parser

    prompt_with_history = ChatPromptTemplate.from_messages([
        ("system", STEP3_PROMPT),
        ("human", "User: {query}\n\n历史任务槽位信息：\n{history_slots}")
    ])
    prompt_with_history = prompt_with_history.partial(format_instructions=parser.get_format_instructions())
    chain_with_history = prompt_with_history | llm | parser

    def _build_history_slots_text(history: dict) -> str:
        """构建历史槽位文本，注入到 prompt 中"""
        lines = []
        if history.get("product_category"):
            lines.append(f"- product_category: {history['product_category']}")
        if history.get("keywords"):
            lines.append(f"- keywords: {history['keywords']}")
        else:
            lines.append("- keywords: []")
        if history.get("filters"):
            lines.append(f"- filters: {history['filters']}")
        else:
            lines.append("- filters: {}")
        return "\n".join(lines)

    results = []
    for item in step2_items:
        if item.intent != IntentCategory.SHOPPING:
            results.append(Step3Item(
                sub_query=item.sub_query,
                intent=item.intent,
                is_new=item.is_new,
                matched_task_id=item.matched_task_id,
            ))
            continue

        # 根据是否有 matched_task_id 选择不同的 prompt 版本
        # matched_slots 的 key 是字符串，与 matched_task_id 类型一致
        if item.matched_task_id and item.matched_task_id in matched_slots:
            # 有历史槽位：注入历史上下文
            history_info = matched_slots[item.matched_task_id]
            history_slots_text = _build_history_slots_text(history_info)
            result = await chain_with_history.ainvoke({
                "query": item.sub_query,
                "history_slots": history_slots_text,
            })
        else:
            # 无历史（新品类或无法匹配）
            result = await chain_no_history.ainvoke({"query": item.sub_query})

        results.append(Step3Item(
            sub_query=item.sub_query,
            intent=item.intent,
            is_new=item.is_new,
            matched_task_id=item.matched_task_id,
            product_category=result.product_category,
            keywords=result.keywords,
            filters=result.filters,
        ))
    return results


# ======================= Step 4: 搜索意愿判断（仅 SHOPPING） =======================

STEP4_PROMPT = """
# Role
你是一个搜索意愿判断专家。根据用户的子查询判断是否要立即触发搜索。

# 判断规则
- **explicit_search=true**: 用户说类似 "搜一下"、"款式随意"、"随便"、"都行"、"直接推荐吧" 的话，总之你能体会到用户想立刻搜索商品的意愿
- **is_replace_products=true**: 用户说"换一批"、"再看看"、"换一个"

# 重要
- "款式随意"、"随便"、"都行" → explicit_search=true（用户不想澄清，要求直接搜）
- "换一批"、"再看看" → is_replace_products=true（用户想看更多商品）
- 两者可以同时为 true

# Output Format
输出标准 JSON 对象，wrap 在 ```json ``` 代码块中。示例：

```json
{{"explicit_search": true, "is_replace_products": false}}
```

# Few-Shot Examples

User: "款式随便，直接推荐吧"
```json
{{"explicit_search": true, "is_replace_products": false}}
```

User: "换一批看看"
```json
{{"explicit_search": false, "is_replace_products": true}}
```

User: "推荐一款降噪耳机，要索尼的"
```json
{{"explicit_search": false, "is_replace_products": false}}
```
"""


async def step4_search_willingness(llm, step3_items: list[Step3Item]) -> list[Step4Item]:
    """Step 4: 搜索意愿判断（仅 SHOPPING）"""
    from pydantic import create_model
    DynamicStep4Item = create_model(
        "DynamicStep4Item",
        explicit_search=(bool, False),
        is_replace_products=(bool, False),
    )
    parser = PydanticOutputParser(pydantic_object=DynamicStep4Item)
    prompt = ChatPromptTemplate.from_messages([
        ("system", STEP4_PROMPT),
        ("human", "User: {query}")
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser

    results = []
    for item in step3_items:
        if item.intent != IntentCategory.SHOPPING:
            results.append(Step4Item(
                sub_query=item.sub_query,
                intent=item.intent,
                is_new=item.is_new,
                matched_task_id=item.matched_task_id,
                explicit_search=False,
                is_replace_products=False,
            ))
            continue

        result = await chain.ainvoke({"query": item.sub_query})
        results.append(Step4Item(
            sub_query=item.sub_query,
            intent=item.intent,
            is_new=item.is_new,
            matched_task_id=item.matched_task_id,
            product_category=item.product_category,
            keywords=item.keywords,
            filters=item.filters,
            explicit_search=result.explicit_search,
            is_replace_products=result.is_replace_products,
        ))
    return results


# ======================= 最终组装 =======================

def _assemble_intent_items(step4_items: list[Step4Item]) -> IntentResponse:
    """将 Step4 输出组装为 IntentResponse"""
    intent_items = []
    for item in step4_items:
        if item.intent == IntentCategory.SHOPPING:
            intent_items.append(IntentItem(
                sub_query=item.sub_query,
                intent=item.intent,
                is_new=item.is_new,
                matched_task_id=item.matched_task_id,
                extracted_slots={
                    "product_category": item.product_category,
                    "keywords": item.keywords,
                    "filters": item.filters,
                },
                explicit_search=item.explicit_search,
                is_replace_products=item.is_replace_products,
            ))
        else:
            intent_items.append(IntentItem(
                sub_query=item.sub_query,
                intent=item.intent,
                is_new=item.is_new,
                matched_task_id=item.matched_task_id,
                extracted_slots={},
                explicit_search=False,
                is_replace_products=False,
            ))
    return IntentResponse(intent_items=intent_items)


# ======================= 主节点 =======================

async def intent_decomposer_node(state: ShopmindAgentState, runtime: Runtime[ShopmindAssistantContext]):
    """意图分解器节点，四步线性流水线"""
    context = runtime.context
    logger.info(f"[intent_decomposer_node] thread_id: {context.thread_id}")

    cfg = get_config().get("configurable", {})
    max_clarification_count = cfg.get(MAX_CLARIFICATION_COUNT, 3)
    max_history_task_count = cfg.get(MAX_HISTORY_TASK_COUNT, 3)

    llm = context.reasoning_llm

    subtasks = state.get("sub_tasks", [])
    filtered_subtasks = _filter_active_subtasks(subtasks, max_count=max_history_task_count)

    # Step 1: 意图分类 + 原子拆分
    step1_items = await step1_intent_classify(llm, state.get("rewritten_query", ""), filtered_subtasks)

    # Step 2: 历史任务匹配
    step2_items = await step2_task_match(llm, step1_items, filtered_subtasks)

    # Step 3: 槽位提取（仅 SHOPPING）
    step3_items = await step3_extract_slots(llm, step2_items, filtered_subtasks)

    # Step 4: 搜索意愿判断（仅 SHOPPING）
    step4_items = await step4_search_willingness(llm, step3_items)

    # 最终组装: Step4 输出 → IntentResponse
    intent_resp = _assemble_intent_items(step4_items)

    # 处理每个 IntentItem → 创建/更新 SubTask
    for intent_item in intent_resp.intent_items:
        logger.info(f"【意图识别结果】- {intent_item}")
        is_new = intent_item.is_new
        matched_subtask = None

        if not is_new and intent_item.matched_task_id:
            for t in subtasks:
                if t.task_id == intent_item.matched_task_id:
                    matched_subtask = t
                    break

        current_subtask = matched_subtask

        if intent_item.intent == IntentCategory.SHOPPING:
            if is_new or not matched_subtask:
                current_subtask = ShoppingSubTask(
                    category=IntentCategory.SHOPPING,
                    sub_query=intent_item.sub_query,
                    status=TaskStatus.NEW,
                    product_category=intent_item.extracted_slots.get("product_category"),
                    keywords=intent_item.extracted_slots.get("keywords", []),
                    filters=intent_item.extracted_slots.get("filters", {}),
                    is_replace_products=intent_item.is_replace_products,
                )
                logger.info(f"[意图识别] - 新建了一个 shopping 意图 task: {current_subtask}")
                subtasks.append(current_subtask)
            else:
                _update_subtask_slots(matched_subtask, intent_item.extracted_slots)
                matched_subtask.is_replace_products = intent_item.is_replace_products
                logger.info(f"[意图识别]-匹配到历史任务 subtask: {matched_subtask}")
                current_subtask = matched_subtask
            _handle_clarification(current_subtask, intent_item, max_clarification_count)

        elif intent_item.intent == IntentCategory.PLATFORM:
            if is_new or not matched_subtask:
                new_subtask = PlatformSubTask(
                    category=IntentCategory.PLATFORM,
                    sub_query=intent_item.sub_query,
                    status=TaskStatus.NEW,
                )
                current_subtask = new_subtask
                subtasks.append(new_subtask)

        elif intent_item.intent == IntentCategory.CHITCHAT:
            if is_new or not matched_subtask:
                new_subtask = ChitchatSubTask(
                    category=IntentCategory.CHITCHAT,
                    sub_query=intent_item.sub_query,
                    status=TaskStatus.NEW,
                )
                current_subtask = new_subtask
                subtasks.append(new_subtask)

        elif intent_item.intent == IntentCategory.COMPARISON:
            product_ids = []
            if matched_subtask and isinstance(matched_subtask, ShoppingSubTask):
                product_ids = matched_subtask.has_recommended_product_ids or []

            new_subtask = ComparisonSubTask(
                category=IntentCategory.COMPARISON,
                sub_query=intent_item.sub_query,
                status=TaskStatus.NEW,
                product_ids=product_ids,
            )
            current_subtask = new_subtask
            logger.info(f"[意图识别] - 新建了一个 comparison 意图 task: {current_subtask}")
            subtasks.append(new_subtask)

        state["current_tasks"].append(current_subtask)

    logger.info(f"[意图识别] - thread_id:{context.thread_id}, subtasks 个数: {len(subtasks)}, 当前任务个数: {len(subtasks)}")
    state["sub_tasks"] = subtasks
    return state


def _update_subtask_slots(subtask: SubTask, extracted_slots: dict):
    """更新已有 SubTask 的槽位"""
    if not isinstance(subtask, ShoppingSubTask):
        return
    if extracted_slots.get("product_category") and not subtask.product_category:
        subtask.product_category = extracted_slots.get("product_category")
    new_keywords = extracted_slots.get("keywords", [])
    if new_keywords:
        subtask.keywords = new_keywords
    subtask.filters = extracted_slots.get("filters", {})


def _handle_clarification(subtask: SubTask, intent_item: IntentItem, max_clarification_count: int):
    """判断是否需要澄清"""
    if not isinstance(subtask, ShoppingSubTask):
        subtask.status = TaskStatus.READY
        return
    if intent_item.explicit_search:
        subtask.status = TaskStatus.READY
        return
    if subtask.clarification_count >= max_clarification_count:
        subtask.status = TaskStatus.READY
    else:
        subtask.status = TaskStatus.CLARIFYING
