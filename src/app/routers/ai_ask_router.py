"""AI 对话路由"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.ai_ask_schema import (
    AIAskRequest,
    AIClearHistoryRequest,
    UpdateConversationNameRequest,
    DeleteConversationRequest,
)
from app.schemas.result_context import ResultContext
from app.services.ai_chat_service import get_ai_chat_service
from app.utils.logger import app_logger as logger

router = APIRouter(prefix="/ai/chat", tags=["大模型对话"])


@router.post(
    "",
    summary="大模型对话（流式）",
    description="""
    大模型对话接口，支持流式输出和短期记忆。
    
    参数说明：
    - question: 用户的问题（必填）
    - user_id: 用户ID（可选，用于区分不同用户）
    - session_id: 会话ID（可选，用于支持多会话并存）
        - 如果传递 session_id，则使用独立会话
        - 如果不传，则使用该用户的默认会话
        - 不同的 session_id 对应不同的对话历史，互不影响
    
    返回：
    - 流式输出 AI 回复的文本内容（text/plain 格式）
    """,
)
async def chat(request: AIAskRequest):
    """
    大模型对话（流式输出，带短期记忆）
    """
    try:
        logger.info(
            "收到对话请求",
            extra={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "question": request.question,
            }
        )
        
        # 获取对话服务
        chat_service = get_ai_chat_service()
        
        # 返回流式响应
        return StreamingResponse(
            chat_service.chat_stream(request),
            media_type="text/plain; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"对话请求处理失败: {e}", exc_info=True)
        # 流式输出错误信息
        async def error_stream():
            yield f"抱歉，处理您的请求时出现错误：{str(e)}"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/plain; charset=utf-8",
            status_code=500
        )


@router.get(
    "/history/{session_id}",
    response_model=ResultContext[list[dict]],
    summary="获取对话历史",
    description="""
    获取指定会话的对话历史记录。
    
    参数说明：
    - session_id: 会话ID（必填）
    
    返回：
    - 消息历史列表，格式：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    使用场景：
    - 前端打开对话时，先调用此接口获取历史消息
    - 刷新页面后恢复对话历史
    """,
)
async def get_history(session_id: str) -> ResultContext[list[dict]]:
    """
    获取对话历史
    """
    try:
        logger.info(
            "收到获取历史请求",
            extra={"session_id": session_id}
        )
        
        # 获取对话服务
        chat_service = get_ai_chat_service()
        
        # 获取历史
        messages = await chat_service.get_history(session_id=session_id)
        
        return ResultContext.ok(
            data=messages,
            message="获取对话历史成功"
        )
            
    except Exception as e:
        logger.error(f"获取历史失败: {e}", exc_info=True)
        return ResultContext.fail(
            message=f"获取历史失败: {str(e)}",
            code="GET_HISTORY_ERROR"
        )


@router.post(
    "/clear-history",
    response_model=ResultContext[dict],
    summary="清除对话历史",
    description="""
    清除对话历史记录。
    
    参数说明：
    - session_id: 会话ID（必填）
    
    使用场景：
    - 前端提供"清除对话"按钮，点击后调用此接口
    """,
)
async def clear_history(request: AIClearHistoryRequest) -> ResultContext[dict]:
    """
    清除对话历史
    """
    try:
        logger.info(
            "收到清除历史请求",
            extra={
                "session_id": request.session_id
            }
        )
        
        # 获取对话服务
        chat_service = get_ai_chat_service()
        
        # 清除历史
        success = await chat_service.clear_history(session_id=request.session_id)
        
        if success:
            return ResultContext.ok(
                data={"cleared": True},
                message="对话历史已清除"
            )
        else:
            return ResultContext.fail(
                message="清除对话历史失败",
                code="CLEAR_FAILED"
            )
            
    except Exception as e:
        logger.error(f"清除历史失败: {e}", exc_info=True)
        return ResultContext.fail(
            message=f"清除历史失败: {str(e)}",
            code="CLEAR_ERROR"
        )


# ========== 对话列表管理接口 ==========

@router.get(
    "/conversations/{user_id}",
    response_model=ResultContext[list[dict]],
    summary="获取对话列表",
    description="""
    获取用户的所有对话列表。
    
    参数说明：
    - user_id: 用户ID（必填）
    
    返回：
    - 对话列表，每个元素包含 session_id 和 name
    """,
)
async def get_conversations(user_id: str) -> ResultContext[list[dict]]:
    """
    获取用户的所有对话列表
    """
    try:
        logger.info(
            "收到获取对话列表请求",
            extra={"user_id": user_id}
        )
        
        # 获取对话服务
        chat_service = get_ai_chat_service()
        
        # 获取对话列表
        conversations = await chat_service.get_conversation_list(user_id)
        
        return ResultContext.ok(
            data=conversations,
            message="获取对话列表成功"
        )
            
    except Exception as e:
        logger.error(f"获取对话列表失败: {e}", exc_info=True)
        return ResultContext.fail(
            message=f"获取对话列表失败: {str(e)}",
            code="GET_CONVERSATIONS_ERROR"
        )


@router.post(
    "/conversations/update-name",
    response_model=ResultContext[dict],
    summary="更新对话名称",
    description="""
    新增对话，若有则更新对话名称。
    
    参数说明：
    - user_id: 用户ID（必填）
    - session_id: 会话ID（必填）
    - name: 新对话名称（必填）
    """,
)
async def update_conversation_name(request: UpdateConversationNameRequest) -> ResultContext[dict]:
    """
    更新对话名称
    """
    try:
        logger.info(
            "收到更新对话名称请求",
            extra={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "conversation_name": request.name
            }
        )
        
        # 获取对话服务
        chat_service = get_ai_chat_service()
        
        # 更新对话名称
        success = await chat_service.update_conversation_name(
            request.user_id,
            request.session_id,
            request.name
        )
        
        if success:
            return ResultContext.ok(
                data={"updated": True},
                message="更新对话名称成功"
            )
        else:
            return ResultContext.fail(
                message="更新对话名称失败",
                code="UPDATE_NAME_FAILED"
            )
            
    except Exception as e:
        logger.error(f"更新对话名称失败: {e}", exc_info=True)
        return ResultContext.fail(
            message=f"更新对话名称失败: {str(e)}",
            code="UPDATE_NAME_ERROR"
        )


@router.post(
    "/conversations/delete",
    response_model=ResultContext[dict],
    summary="删除对话",
    description="""
    删除对话（包括对话历史）。
    
    参数说明：
    - user_id: 用户ID（必填）
    - session_id: 会话ID（必填）
    """,
)
async def delete_conversation(request: DeleteConversationRequest) -> ResultContext[dict]:
    """
    删除对话
    """
    try:
        logger.info(
            "收到删除对话请求",
            extra={
                "user_id": request.user_id,
                "session_id": request.session_id
            }
        )
        
        # 获取对话服务
        chat_service = get_ai_chat_service()
        
        # 删除对话
        success = await chat_service.delete_conversation(
            request.user_id,
            request.session_id
        )
        
        if success:
            return ResultContext.ok(
                data={"deleted": True},
                message="删除对话成功"
            )
        else:
            return ResultContext.fail(
                message="删除对话失败",
                code="DELETE_FAILED"
            )
            
    except Exception as e:
        logger.error(f"删除对话失败: {e}", exc_info=True)
        return ResultContext.fail(
            message=f"删除对话失败: {str(e)}",
            code="DELETE_ERROR"
        )