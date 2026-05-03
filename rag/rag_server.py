"""
RAG API 服务
提供嵌入生成和相似检索功能，与 Agent 主程序分离

启动命令:
    python rag_api.py --port 8000

API 端点:
    POST /embed      - 生成嵌入向量
    POST /retrieve   - 检索相似案例
    GET  /health     - 健康检查

#多卡
CUDA_VISIBLE_DEVICES=6 python /local_data/lizhijian/Envi_KnowGraph/rag/rag_server.py \
    --port 8008 \
    --model_path /local_data/lizhijian/model_weight/QwenVL_Embedding \
    --embedding_lib_dir /local_data/lizhijian/Envi_KnowGraph/rag/embedding_lib \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.04

#单卡
CUDA_VISIBLE_DEVICES=6 python rag/rag_server.py \
    --port 8008 \
    --model_path /local_data/lizhijian/model_weight/QwenVL_Embedding \
    --embedding_lib_dir /local_data/lizhijian/WS/DisasterAgent/rag/embedding_lib \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.25
    
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Any
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import argparse

# ============================================================
# 配置 - 适配 embedding_lib 文件夹结构
# ============================================================
DEFAULT_MODEL_PATH = "/local_data/lizhijian/model_weight/QwenVL_Embedding"
DEFAULT_EMBEDDING_LIB_DIR = "/local_data/lizhijian/RAG/rag/embedding_lib"

# ============================================================
# 请求/响应模型
# ============================================================
class EmbedRequest(BaseModel):
    pre_image_path: str
    post_image_path: str
    disaster_type: str
    instruction: Optional[str] = "Represent the disaster scene for retrieval."

class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int

class RetrieveRequest(BaseModel):
    pre_image_path: str
    post_image_path: str
    disaster_type: str
    top_k: int = 3

class RetrieveResponse(BaseModel):
    results: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    embeddings_loaded: bool
    total_embeddings: int

# ============================================================
# RAG 服务类
# ============================================================
class RAGService:
    def __init__(
        self,
        model_path: str,
        embedding_lib_dir: str,
        tensor_parallel_size: int = 8,
        gpu_memory_utilization: float = 0.08,
    ):
        self.model_path = model_path
        self.embedding_lib_dir = embedding_lib_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        
        self.llm = None
        self.embedding_ids: List[str] = []
        self.embedding_vectors: Optional[np.ndarray] = None
        self.retrieval_targets: Dict[str, Dict] = {}
        
    def load_model(self):
        """加载嵌入模型"""
        print(f"正在加载模型: {self.model_path}")
        from vllm import LLM
        
        self.llm = LLM(
            model=self.model_path,
            runner="pooling",
            dtype='bfloat16',
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=8192,
            max_num_seqs=1,
        )
        print("模型加载完成!")
        
    def load_embeddings(self):
        """加载预计算的嵌入向量库"""
        print(f"正在加载嵌入向量库: {self.embedding_lib_dir}")
        
        # 加载嵌入向量
        vectors_path = os.path.join(self.embedding_lib_dir, "embedding_vectors.npy")
        ids_path = os.path.join(self.embedding_lib_dir, "embedding_ids.json")
        targets_path = os.path.join(self.embedding_lib_dir, "rag_retrieval_targets.json")
        
        # 加载向量
        if os.path.exists(vectors_path):
            self.embedding_vectors = np.load(vectors_path)
            print(f"  嵌入向量形状: {self.embedding_vectors.shape}")
        else:
            raise FileNotFoundError(f"找不到嵌入向量文件: {vectors_path}")
        
        # 加载 ID 列表
        if os.path.exists(ids_path):
            with open(ids_path, 'r', encoding='utf-8') as f:
                self.embedding_ids = json.load(f)
            print(f"  嵌入 ID 数量: {len(self.embedding_ids)}")
        else:
            raise FileNotFoundError(f"找不到 ID 文件: {ids_path}")
        
        # 加载检索目标
        if os.path.exists(targets_path):
            with open(targets_path, 'r', encoding='utf-8') as f:
                self.retrieval_targets = json.load(f)
            print(f"  检索目标数量: {len(self.retrieval_targets)}")
        else:
            raise FileNotFoundError(f"找不到检索目标文件: {targets_path}")
        
        # 预计算归一化向量（加速检索）
        self.embedding_vectors_norm = self.embedding_vectors / np.linalg.norm(
            self.embedding_vectors, axis=1, keepdims=True
        )
        print("  嵌入向量归一化完成")
        
    def _prepare_input(self, pre_image_path: str, post_image_path: str, 
                       disaster_type: str, instruction: str) -> List[Dict]:
        """准备模型输入"""
        content = []
        
        # 添加图像
        for img_path in [pre_image_path, post_image_path]:
            if img_path.startswith(('http://', 'https://')):
                image_content = img_path
            else:
                abs_path = os.path.abspath(img_path)
                image_content = 'file://' + abs_path
            content.append({'type': 'image', 'image': image_content})
        
        # 添加文本
        content.append({
            'type': 'text', 
            'text': f"Disaster type: {disaster_type}. Compare the pre-disaster and post-disaster satellite images."
        })
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}
        ]
        
        return conversation
    
    def _prepare_vllm_input(self, conversation: List[Dict], 
                            pre_image_path: str, post_image_path: str) -> Dict:
        """准备 vLLM 输入格式"""
        prompt_text = self.llm.llm_engine.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 加载图像
        image_objects = []
        for img_path in [pre_image_path, post_image_path]:
            if img_path.startswith(('http://', 'https://')):
                from vllm.multimodal.utils import fetch_image
                image_obj = fetch_image(img_path)
            else:
                abs_path = os.path.abspath(img_path)
                image_obj = Image.open(abs_path).convert("RGB")
            image_objects.append(image_obj)
        
        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image_objects}
        }
    
    def embed(self, pre_image_path: str, post_image_path: str, 
              disaster_type: str, instruction: str = "Represent the disaster scene for retrieval.") -> np.ndarray:
        """生成嵌入向量"""
        if self.llm is None:
            raise RuntimeError("模型未加载")
        
        conversation = self._prepare_input(pre_image_path, post_image_path, disaster_type, instruction)
        vllm_input = self._prepare_vllm_input(conversation, pre_image_path, post_image_path)
        
        outputs = self.llm.embed([vllm_input])
        embedding = np.array(outputs[0].outputs.embedding)
        
        return embedding
    
    def retrieve(self, pre_image_path: str, post_image_path: str, 
                 disaster_type: str, top_k: int = 3) -> List[Dict]:
        """
        检索相似案例
        
        策略：在全局向量库中检索，然后可选按灾害类型过滤
        """
        # 生成查询嵌入
        query_embedding = self.embed(pre_image_path, post_image_path, disaster_type)
        
        # 归一化查询向量
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # 计算余弦相似度
        similarities = np.dot(self.embedding_vectors_norm, query_norm)
        
        # 可选：按灾害类型过滤
        # 找出匹配灾害类型的索引
        matching_indices = []
        for i, sample_id in enumerate(self.embedding_ids):
            # ID 格式: "earthquake_000", "fire_001" 等
            id_type = sample_id.rsplit('_', 1)[0]
            if id_type == disaster_type:
                matching_indices.append(i)
        
        if matching_indices:
            # 只在匹配类型中检索
            filtered_similarities = similarities[matching_indices]
            top_local_indices = np.argsort(filtered_similarities)[::-1][:top_k]
            top_indices = [matching_indices[i] for i in top_local_indices]
        else:
            # 如果没有匹配类型，在全局中检索
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 构建结果
        results = []
        for idx in top_indices:
            sample_id = self.embedding_ids[idx]
            target = self.retrieval_targets.get(sample_id, {})
            results.append({
                "id": sample_id,
                "similarity": float(similarities[idx]),
                "caption_answer": target.get("caption_answer", ""),
                "advice_answer": target.get("advice_answer", "")
            })
        
        return results

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="RAG API", description="灾害场景嵌入与检索服务")
rag_service: Optional[RAGService] = None

@app.on_event("startup")
async def startup():
    """启动时加载模型和嵌入"""
    global rag_service
    
    model_path = os.environ.get("RAG_MODEL_PATH", DEFAULT_MODEL_PATH)
    embedding_lib_dir = os.environ.get("RAG_EMBEDDING_LIB_DIR", DEFAULT_EMBEDDING_LIB_DIR)
    tp_size = int(os.environ.get("RAG_TP_SIZE", "8"))
    gpu_util = float(os.environ.get("RAG_GPU_UTIL", "0.08"))
    
    rag_service = RAGService(
        model_path=model_path,
        embedding_lib_dir=embedding_lib_dir,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_util,
    )
    rag_service.load_embeddings()
    rag_service.load_model()

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(
        status="ok",
        model_loaded=rag_service is not None and rag_service.llm is not None,
        embeddings_loaded=rag_service is not None and rag_service.embedding_vectors is not None,
        total_embeddings=len(rag_service.embedding_ids) if rag_service else 0
    )

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """生成嵌入向量"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        embedding = rag_service.embed(
            request.pre_image_path,
            request.post_image_path,
            request.disaster_type,
            request.instruction
        )
        return EmbedResponse(
            embedding=embedding.tolist(),
            dimension=len(embedding)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """检索相似案例"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        results = rag_service.retrieve(
            request.pre_image_path,
            request.post_image_path,
            request.disaster_type,
            request.top_k
        )
        return RetrieveResponse(results=results)
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] /retrieve 请求失败:\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{error_detail}")

# ============================================================
# 客户端封装（供 Agent 调用）
# ============================================================
class RAGClient:
    """RAG API 客户端，供 Agent 主程序调用"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        
    def embed(self, pre_image_path: str, post_image_path: str, 
              disaster_type: str, instruction: str = "Represent the disaster scene for retrieval.") -> List[float]:
        """生成嵌入向量"""
        import requests
        
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
    
    def retrieve(self, pre_image_path: str, post_image_path: str, 
                 disaster_type: str, top_k: int = 3) -> List[Dict]:
        """检索相似案例"""
        import requests
        
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
    
    def health(self) -> Dict:
        """健康检查"""
        import requests
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG API 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="嵌入模型路径")
    parser.add_argument("--embedding_lib_dir", type=str, default=DEFAULT_EMBEDDING_LIB_DIR,
                        help="嵌入向量库目录 (包含 embedding_vectors.npy, embedding_ids.json, rag_retrieval_targets.json)")
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                        help="张量并行大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.08,
                        help="GPU 显存使用率")
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["RAG_MODEL_PATH"] = args.model_path
    os.environ["RAG_EMBEDDING_LIB_DIR"] = args.embedding_lib_dir
    os.environ["RAG_TP_SIZE"] = str(args.tensor_parallel_size)
    os.environ["RAG_GPU_UTIL"] = str(args.gpu_memory_utilization)
    
    # 启动服务
    print("=" * 60)
    print("RAG API 服务")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"嵌入库目录: {args.embedding_lib_dir}")
    print(f"张量并行: {args.tensor_parallel_size}")
    print(f"显存使用率: {args.gpu_memory_utilization}")
    print(f"监听地址: {args.host}:{args.port}")
    print("=" * 60)
    
    uvicorn.run(app, host=args.host, port=args.port)
