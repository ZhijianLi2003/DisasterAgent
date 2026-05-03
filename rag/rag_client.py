"""
RAG API 客户端
供 Agent 主程序调用，与 RAG API 服务通信

使用示例:
    from rag_client import RAGClient
    
    client = RAGClient(base_url="http://localhost:8008")
    
    # 检索相似案例
    results = client.retrieve(
        pre_image_path="/path/to/pre.png",
        post_image_path="/path/to/post.png",
        disaster_type="earthquake",
        top_k=3
    )
"""

import requests
from typing import List, Dict, Optional


class RAGClient:
    """RAG API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8008"):
        """
        初始化客户端
        
        Args:
            base_url: RAG API 服务地址
        """
        self.base_url = base_url.rstrip("/")
    
    def health(self) -> Dict:
        """
        健康检查
        
        Returns:
            包含服务状态的字典
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def embed(
        self, 
        pre_image_path: str, 
        post_image_path: str, 
        disaster_type: str, 
        instruction: str = "Represent the disaster scene for retrieval."
    ) -> List[float]:
        """
        生成嵌入向量
        
        Args:
            pre_image_path: 灾前图像路径
            post_image_path: 灾后图像路径
            disaster_type: 灾害类型
            instruction: 嵌入指令
            
        Returns:
            嵌入向量 (List[float])
        """
        response = requests.post(
            f"{self.base_url}/embed",
            json={
                "pre_image_path": pre_image_path,
                "post_image_path": post_image_path,
                "disaster_type": disaster_type,
                "instruction": instruction
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    def retrieve(
        self, 
        pre_image_path: str, 
        post_image_path: str, 
        disaster_type: str, 
        top_k: int = 3
    ) -> List[Dict]:
        """
        检索相似案例
        
        Args:
            pre_image_path: 灾前图像路径
            post_image_path: 灾后图像路径
            disaster_type: 灾害类型 (earthquake/explosion/fire/flood/hurricane/tornado/tsunami/volcano)
            top_k: 返回前 K 个相似案例
            
        Returns:
            相似案例列表，每个案例包含:
            - id: 案例 ID
            - similarity: 相似度分数
            - caption_answer: 灾害描述
            - advice_answer: 恢复建议
        """
        response = requests.post(
            f"{self.base_url}/retrieve",
            json={
                "pre_image_path": pre_image_path,
                "post_image_path": post_image_path,
                "disaster_type": disaster_type,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        return response.json()["results"]
    
    def is_available(self) -> bool:
        """
        检查服务是否可用
        
        Returns:
            True 如果服务可用，否则 False
        """
        try:
            result = self.health()
            return result.get("status") == "ok" and result.get("model_loaded", False)
        except Exception:
            return False


# 便捷函数
def create_client(base_url: str = "http://localhost:8008") -> RAGClient:
    """创建 RAG 客户端"""
    return RAGClient(base_url=base_url)
