# Resume-safe project wording

The evaluation uses simulated profiles because the source catalog has no real
user histories. Keep that qualifier whenever quoting personalization results.

## English

- Built a multimodal semantic product-discovery system over 44,419 fashion
  products using CLIP and FAISS, supporting text/image retrieval through a
  FastAPI ML service and Spring Boot orchestration.
- Designed PostgreSQL-backed user profiles from weighted behavioral signals and
  implemented a hybrid reranker combining query similarity, semantic user
  preference, and category/color/usage preferences.
- Created a reproducible 72-query retrieval benchmark; CLIP improved overall
  NDCG@10 by 24.3% over BM25 (0.508 to 0.631) and substantially outperformed
  lexical retrieval on paraphrased queries.
- Ran a 171-persona simulated-user ablation study in which the full hybrid
  reranker improved NDCG@10 from 0.032 to 0.405 versus non-personalized CLIP;
  reported paired-bootstrap confidence intervals and explicit data limitations.

## 中文

- 基于 CLIP 与 FAISS 构建覆盖 44,419 件时尚商品的多模态语义检索系统，支持
  文字与图片搜索，并采用 FastAPI 模型服务和 Spring Boot 业务编排架构。
- 根据加权用户行为构建 PostgreSQL 用户画像，实现融合查询相似度、用户语义
  向量以及品类、颜色和使用场景偏好的混合重排序。
- 建立包含 72 条查询的可重复检索基准；CLIP 相比 BM25 将整体 NDCG@10 从
  0.508 提升至 0.631（+24.3%），并在同义改写查询上显著优于关键词检索。
- 在 171 个模拟用户画像的消融实验中，完整混合排序相较非个性化 CLIP 将
  NDCG@10 从 0.032 提升至 0.405，并报告配对 Bootstrap 置信区间与数据限制。
