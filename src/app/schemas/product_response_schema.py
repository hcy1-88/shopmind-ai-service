"""
@File       : product_response_schema.py
@Description:

@Time       : 2026/1/6 12:55
@Author     : hcy18
"""
from decimal import Decimal
from typing import Optional, Union
from pydantic import Field, field_serializer
from app.schemas.base import CamelCaseModel

class PriceRange(CamelCaseModel):
    """价格范围."""
    min: Optional[Decimal] = Field(default=None, description="最低价格")
    max: Optional[Decimal] = Field(default=None, description="最高价格")


class ProductSkuResponseDto(CamelCaseModel):
    """商品 SKU 响应 DTO（规格组合）."""
    id: Union[int, str] = Field(..., description="SKU ID")
    attributes: Optional[dict[str, str]] = Field(default=None, description="规格属性，如 {\"颜色\": \"星光色\", \"存储\": \"512GB\"}")
    price: Optional[Decimal] = Field(default=None, description="SKU 价格")
    stock: Optional[int] = Field(default=None, description="库存")
    image: Optional[str] = Field(default=None, description="SKU 图片 URL")


class TagInfo(CamelCaseModel):
    """标签信息."""
    name: str = Field(..., description="标签名称")
    type: Optional[str] = Field(None, description="标签类型")


class ProductResponseDto(CamelCaseModel):
    """
    商品响应 DTO（与 Java 服务保持一致）.

    Example Json:
    {
		"id": "264498826428022784",
		"name": "Apple苹果笔记本电脑MacBookPro13寸15办公设计Air超薄",
		"price": 3000,
		"priceRange": {
			"min": 3000,
			"max": 5000
		},
		"image": "http://127.0.0.1:9000/product-service/products/264498827476598784.jpeg",
		"aiSummary": "Apple MacBook Pro 13/15/16寸笔记本电脑，极致轻薄设计，重量仅1.5kg至2kg，便于携带。搭载Intel i5/i7处理器，8GB内存与256GB固态硬盘，显著提升办公与设计效率。高清Retina屏幕，60Hz刷新率，提供出色视觉体验，适合视频剪辑和图像处理等专业需求。支持双系统切换，兼顾工作与娱乐。购买享1年保修及500元优惠券，是应对各种办公和设计挑战的超值选择。",
		"description": "Apple MacBook Pro 13/15/16寸笔记本电脑，极致轻薄设计，重量仅1.5kg至2kg，便携性出色。搭载Intel i5/i7处理器，8GB内存与256GB固态硬盘，大幅提升办公和设计效率。高清Retina屏幕，60Hz刷新率，视觉体验更佳，适合视频剪辑、图像处理等专业需求。支持双系统切换，兼顾工作与娱乐。提供1年保修服务，购买即享500元优惠券，超值选择，助您轻松应对各种办公和设计挑战。",
		"merchantId": "2001648668219994113",
		"location": "北京市 市辖区 西城区",
		"category": "100002",
		"tagInfo": [
			{
				"name": "极致轻薄",
				"color": "#1E90FF"
			},
			{
				"name": "高清Retina屏",
				"color": "#1E90FF"
			},
			{
				"name": "双系统切换",
				"color": "#1E90FF"
			},
			{
				"name": "适合设计",
				"color": "#32CD32"
			},
			{
				"name": "500元优惠",
				"color": "#FF4500"
			}
		],
		"salesCount": 0,
		"viewCount": 0,
		"likeCount": 0
	}
    """
    id: Union[int, str] = Field(..., description="商品ID（传给前端时为字符串）")
    name: str = Field(..., description="商品名称")
    price: Optional[Decimal] = Field(default=None, description="价格")
    original_price: Optional[Decimal] = Field(default=None, description="原价")
    price_range: Optional[PriceRange] = Field(default=None, description="价格范围")
    image: Optional[str] = Field(default=None, description="预览图/封面")
    images: Optional[list[str]] = Field(default_factory=list, description="详情图列表")
    ai_summary: Optional[str] = Field(default=None, description="商品摘要（AI生成）")
    description: Optional[str] = Field(default=None, description="商品描述")
    location: Optional[str] = Field(default=None, description="位置")
    category: Optional[int] = Field(default=None, description="分类ID")
    tag_info: Optional[list[TagInfo]] = Field(default_factory=list, description="商品标签")
    sales_count: Optional[int] = Field(default=None, description="销量")
    view_count: Optional[int] = Field(default=None, description="浏览量")
    like_count: Optional[int] = Field(default=None, description="点赞量")
    skus: Optional[list[ProductSkuResponseDto]] = Field(default_factory=list, description="商品规格组合")


    @field_serializer('id')
    def serialize_id(self, value: Union[int, str]) -> str:
        """将 id 序列化为字符串（传给前端）."""
        return str(value)

    @property
    def id_int(self) -> int:
        """获取整数类型的 id（用于内部处理）."""
        return int(self.id) if isinstance(self.id, str) else self.id

    class Config:
        populate_by_name = True