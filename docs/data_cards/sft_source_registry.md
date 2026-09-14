# SFT 候选数据源登记卡

状态：**Phase 3/4 内部快速版已批准**。实际文件从运行参数传入，registry 不保存机器绝对路径；公开发布仍为 `release_allowed=false`。

| source_id | revision | 上游许可证声明 | 本项目状态 | 规模 | 结论 |
|---|---|---|---|---:|---|
| `shibing624_medical_6e219f1` | `6e219f1a14856833ee436063d3b73c5f1ab9cfb9` | Apache-2.0 | 内部候选用途 approved | 1,950,972 | 仅 6 个无禁止级 overlap 的原始问法进入 validation；不使用原答案，重新绑定权威证据。 |
| `map_coig_cqia_8b55868` | `8b55868c6168adf86c30e7ca0f782cca1c514297` | unknown | pending | 44,694 | 只用于任务分类与难度参考，不复制其文本，不进入冻结集。 |
| `project_authored_eval_v1` | `phase3-fast-v1` | internal-evaluation-only | approved | 174 | 独立编写，权威证据审核；仅限内部评测。 |
| `databricks_dolly_15k_bdd27f4` | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | CC-BY-SA-3.0 | training candidate approved | 15,011 | Phase 4 仅选用非医疗、无联系方式样式且无冻结重叠的通用记录。 |
| `project_authored_training_v1` | `phase4-fast-v1` | internal-training-only | approved | 12,100 候选 | 独立编写并执行代理证据审核；仅限内部工程训练。 |

medical 三个原始 split 的文件 SHA256 分别为 train `98f111016800f98ef4e5c5e29f9260c8ce8ef4331897523d6c3a61003957fed3`、validation `397ce9692f0f5782227781ffc2cd9cbb18c105e03817000aedc97195584e6992`、test `c3876d82565cb9958be72883a096311a5081fb6704f8607ec043921baf57b1a9`。COIG 文件 SHA256 为 `f2a6a758362b7fa5d5c29c9ae0946f2fbf8c9318b6fb48c0e62f8b774d8ff518`。

上游数据卡只能证明声明内容。冻结的 180 条案例均明确 `contains_personal_data=false`，医疗案例绑定 evidence registry，并由 `codex_research_reviewer` 标记为 `agent_evidence_reviewed`；这不是医学专家审核。旧项目的 analysis、eval、experiments、corrective safety、模型输出和 DPO 文件均未作为输入。

Phase 4 的训练批准独立于 Phase 3 评测批准。COIG 许可证仍为 pending，不作为训练文本；Phase 4 正式数据的审核状态仅代表内部工程审核，不是医生审核或临床验证。
