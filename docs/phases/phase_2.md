# Phase 2：RAG 兼容性验收

状态：**已完成**。32 条冻结 query 的新旧检索结果全部满足兼容性门禁；Phase 3 保持锁定。

## 输入与环境

- Phase 1 migration manifest 与 RAG artifact manifest v2 均重新通过实际文件哈希和结构校验。
- query 集为 `data/rag/compatibility/queries.v1.jsonl`，SHA256 为 `b0258d11033e0018d0d2f4fda20d7eb722e63c841344002a7e7458436e416f79`。它由正式 metadata 中 32 个固定、均匀分布的 row 生成，不读取 validation、test 或医疗安全 holdout。
- 新旧链使用同一只读本地 BGE-M3 目录、同一 Python 环境、FlagEmbedding 1.4.2、CPU、FP32、dense-only、max_length 1024 和显式 L2 normalization。
- upstream model revision 无法可靠确认，继续记为 `unknown`；模型权重、配置和 tokenizer 文件指纹为 `913e844cbb77cbe988d0ffe0d27c38e80e35e28e67aebff345bbb88abdd47f16`。

## 验收方法与结果

- 静态层：新旧 metadata 字节一致且均无空行；embedding、index、数量、维度、dtype、norm、Inner Product metric 和 row ID 语义一致。
- 向量层：相同 query 分别执行两次真实 BGE-M3 CPU FP32 编码，再使用旧、新 normalization；最大向量绝对误差为 `0.0`。
- 检索层：每条 query 比较 top-k `1/3/5/10`；全部 row ID、chunk ID、文本、metadata 和排序一致，最大 score 绝对误差为 `0.0`，低于 `1e-5` 门限。
- tie 检查没有发现 `1e-7` 内相邻分数 tie。前 3 条 query 的旧、新公开 retriever 端到端 smoke 全部通过。
- 正式验收 manifest 为 `manifests/phase_2/rag_compatibility.json`，SHA256 为 `09e855f663e3171d662a038044a09f8ddf689dc9bc37ac1369c5f0b1c37d4c46`，不包含机器绝对路径。

## 测试与修复

- Python compileall、ruff 和全部 Phase 0/1/2 测试通过。
- 当前基础环境：24 passed，2 个可选依赖测试跳过；完整 RAG 依赖环境：26 passed，无跳过。
- 首次旧模块动态加载触发 dataclass 模块注册错误；最小修复是在执行旧模块前将其注册到 `sys.modules`。未修改旧项目或检索算法。

## 风险与未执行操作

- 本次结果证明同一组本地模型文件、静态 embedding/index 和运行环境下的新旧工程兼容，不是医疗效果或临床准确率。
- 未改变 chunking，未重新生成 document embedding，未重建正式 FAISS index，未运行 Qwen、RAG generation、训练、SFT、holdout 或 DPO，未进入 Phase 3。
