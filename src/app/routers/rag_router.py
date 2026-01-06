from fastapi import APIRouter, UploadFile, File, Form
from app.schemas.result_context import ResultContext
from app.services.rag_service import get_rag_service
from app.utils.logger import app_logger as logger


router = APIRouter(prefix="/rag", tags=["RAG知识库"])


@router.post(
    "/upload",
    response_model=ResultContext[dict],
    summary="上传文档",
    description="上传文档到知识库，支持 txt、pdf、docx 等格式"
)
async def upload_document(
    file: UploadFile = File(..., description="上传的文件"),
    category: str = Form("platform_rule", description="文档分类")
) -> ResultContext[dict]:
    """上传文档到知识库"""
    try:
        logger.info(f"收到文档上传请求: {file.filename}, 分类: {category}")
        
        # 读取文件内容
        content = await file.read()
        
        # 获取 RAG 服务并上传
        rag_service = get_rag_service()
        result = await rag_service.upload_document(
            file_content=content,
            filename=file.filename or "unknown",
            category=category
        )
        
        return ResultContext.ok(
            data=result,
            message="文档上传成功"
        )
    except Exception as e:
        logger.error(f"文档上传失败: {e}", exc_info=True)
        return ResultContext.fail(
            message=f"文档上传失败: {str(e)}",
            code="UPLOAD_ERROR"
        )


@router.get(
    "/documents",
    response_model=ResultContext[list[dict]],
    summary="获取文档列表",
    description="获取知识库中的所有文档"
)
async def list_documents() -> ResultContext[list[dict]]:
    """获取文档列表"""
    try:
        logger.info("收到文档列表查询请求")
        
        rag_service = get_rag_service()
        documents = await rag_service.list_documents()
        
        return ResultContext.ok(
            data=documents,
            message="查询成功"
        )
    except Exception as e:
        logger.error(f"查询文档列表失败: {e}", exc_info=True)
        return ResultContext.fail(
            message=f"查询失败: {str(e)}",
            code="LIST_ERROR"
        )


@router.delete(
    "/documents/{filename}",
    response_model=ResultContext[dict],
    summary="删除文档",
    description="从知识库中删除指定文档（根据文件名）"
)
async def delete_document(filename: str) -> ResultContext[dict]:
    """删除文档"""
    try:
        logger.info(f"收到文档删除请求: {filename}")
        
        rag_service = get_rag_service()
        success = await rag_service.delete_document(filename)
        
        if success:
            return ResultContext.ok(
                data={"deleted": True, "filename": filename},
                message="文档删除成功"
            )
        else:
            return ResultContext.fail(
                message="文档不存在",
                code="NOT_FOUND"
            )
    except Exception as e:
        logger.error(f"删除文档失败: {e}", exc_info=True)
        return ResultContext.fail(
            message=f"删除失败: {str(e)}",
            code="DELETE_ERROR"
        )
