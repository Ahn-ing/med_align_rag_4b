# Phase 1：RAG 代码及产物迁移

状态：**已完成**。用户批准 corrected cleaned SHA256 后，正式数据迁移和全部验收通过；Phase 2 保持锁定。

## 输入与输出

输入是旧项目中获批的九个 RAG 模块及 Huatuo Encyclopedia 静态 RAG 产物。输出包括 `med_align_rag_4b.rag` 包、迁移后的只读数据目录、migration manifest 和 RAG artifact manifest v2。旧项目始终只读。

## 数据契约与必要改造

- cleaned 记录为稳定的 `id/question/answer`；chunk 保持原有问题前缀、句子切分、768 token 上限、100 token overlap、稳定顺序和 chunk ID。
- 文档与 query embedding 均为 float32 并进行 L2 normalization；检索只使用 `IndexFlatIP`/Inner Product。
- `embeddings.npy` row i、`metadata.jsonl` 第 i 条非空记录和 FAISS row id i 严格对应。
- 所有路径和本地模型目录由调用参数传入；import 不加载模型、不访问网络、不写文件。Qwen 逻辑标识为 `Qwen/Qwen3-4B`，默认 `enable_thinking=false`。
- 清洗、chunk、embedding 和索引发布默认拒绝覆盖并使用同目录临时文件。旧 chunk 输入检查中错误要求 `text` 的问题已改为校验 `id/question/answer`，chunk 算法未改变。

## 迁移文件与 manifest

- 代码迁移详情及源/目标 SHA256：`manifests/phase_1/migration.json`。
- migration manifest：`manifests/phase_1/migration.json`，审核状态为 `approved_and_verified`。
- 正式 artifact manifest：`manifests/rag/huatuo_encyclopedia.v2.json`；数据位于 `data/rag/artifacts/huatuo_encyclopedia/`。

## 验收结果

- Python compileall 与 ruff：通过。
- 当前环境：17 passed，2 个依赖型 FAISS/torch 测试按设计跳过。
- 含完整旧 RAG 依赖的隔离环境：19 passed，无跳过。
- cleaned 361,092 条；chunks 和 metadata 各 377,165 条，二者 SHA256 相同。
- embeddings shape `(377165, 1024)`、dtype float32；全量 norm 范围 `[0.9999998807907104, 1.0000001192092896]`，超出 `1 ± 1e-3` 的记录为 0。
- FAISS 为 `IndexFlatIP`，metric 为 Inner Product，ntotal 377,165，dimension 1,024。正式目录六项历史文件 SHA256 均与源文件一致。

## 风险与未执行操作

- 历史资料无法可靠确认 BGE-M3 精确 revision，正式 manifest 写为 `null`；数据来源许可证记录也为 `unknown`，后续公开发布前必须补齐。
- 用户已批准实际 index SHA256 `67380c9841ff788698c7f017536519467ee835a2a1e9d03240ee7f24a8a050f7`；其索引向量与 embeddings 已逐元素确认一致。
- 请求给出的 cleaned SHA256 是 65 位；用户随后明确批准实际 64 位值 `763b07cfa80964279673d36cde76c8542226cc82daa30ca9c882e34873a22d21`，该值已写入正式 manifest。
- 未下载或加载 Qwen3-4B，未重新生成 embedding，未运行真实 generator、训练、SFT evaluation、安全 holdout 或 DPO，未执行 Phase 2 query 兼容性评测。
