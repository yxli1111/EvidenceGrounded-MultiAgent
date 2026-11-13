# D：证据核验与统一改写

按提供的 `PROJECT_CONTRACT.md` v0.1 实现。三个业务接口不变，纯 Python 标准库，无新增第三方依赖。模型、数据、检索和正式实验由团队现有模块提供。

## 先运行离线版本

在本目录执行：

```powershell
python -m unittest discover -s tests_d -v
python -m examples.demo_d --output demo_result.json
```

Demo 对 M3/M4/M5 使用同一份初稿，调用真正的 D 函数；A 的草稿/批评、B 的证据、C 的 claims/模型响应使用固定替身。测试与 demo 无网络调用，不需要 API key。输出明确标记为 scripted mock，不是模型效果或研究结果。

## 合入项目的文件

| 路径 | 用途 |
| --- | --- |
| `src/verification/` | M4/M5 核验、校验、统一配置和共享类型适配 |
| `src/agents/revision.py` | M3/M4/M5 唯一的改写实现 |
| `prompts/verification/` | 两种核验提示词 |
| `prompts/revision/` | 共用改写提示词 |
| `tests_d/` | D 的离线单元测试及显式 mock |
| `examples/demo_d.py` | 一条合成样本的离线接线示例 |
| `research/D_revision_safety.ipynb` | 使用 E 独立评分的修改分析模板，未填入研究结果 |

保留这些相对路径；提示词按项目根目录定位。没有创建生产 `contracts/`、`src/core/`、检索器或实验 runner。`src/` 与 `src/agents/` 使用 Python namespace package，合入已有同名目录时不覆盖团队的包初始化文件。

## E 的接入方式

```python
from src.verification import configure, runtime_metadata, verify_draft, verify_claims
from src.agents.revision import revise_answer

# 开始一个实验时配置一次，使用团队冻结的模型和参数。
configure(model=experiment_model, temperature=0.0,
          verification_max_tokens=2048, revision_max_tokens=2048)

# M3：A 返回 label=None、evidence_ids=[] 的批评。
m3 = revise_answer(example, shared_draft, a_feedback, llm)

# M4：B 返回整段解释的检索证据。
feedback4 = verify_draft(example, shared_draft, draft_evidence, llm)
m4 = revise_answer(example, shared_draft, feedback4, llm)

# M5：C 返回 claims，B 返回每条 claim 的证据。
feedback5 = verify_claims(example, shared_draft, claims, evidence_by_claim, llm)
m5 = revise_answer(example, shared_draft, feedback5, llm)

d_metadata = runtime_metadata()  # 加入 E 的 ExperimentRecord
```

`example`、`shared_draft`、`claims`、`evidence` 和 `feedback` 是合同中的 dataclass 对象。`evidence_by_claim` 必须包含每个 claim_id；没有检索结果时显式传 `[]`，缺少或多出 ID 会报错。

`llm` 只需实现合同中的 `generate(stage, messages, model, temperature, max_tokens)`，返回有 `.text` 字段的 LLMResponse。所有调用都经这个注入对象完成；没有直接访问任何提供商。C 负责传输重试、缓存、usage 和原始请求/响应日志。D 使用的 stage 为 `verify_draft`、`verify_claims`、`revise_answer`。

M3/M4/M5 的改写代码、系统提示词和模型参数完全共用；不要在不同条件之间调用 `configure` 切换配置。每次 `revise_answer` 最多发起一次 `generate`，E 负责每个方法最多调用一次该函数。发生模型输出错误时 D 不再发起修复改写。

## 共享类型边界

提供的文件中没有真实的 `contracts/schemas.py`。生产代码在 `src/verification/schema_adapter.py` 使用真正的 `VerificationFeedback`、`RevisionResult` 构造返回值，不会自动退回 mock。

合同没有规定反馈 item 的类名，适配器从 `VerificationFeedback.items` 的类型注解取得该 dataclass。要求它包含：`scope`、`target_id`、`rationale`、`label`、`evidence_ids`、可选的 `suggested_correction`。如果团队最终类型有不同的包装方式，仅对齐这个适配文件即可，业务函数签名不变。

Mock schemas 只在 `tests_d.support.mock_contracts()` 上下文中临时注入，退出后恢复；没有写入项目根目录。真实接入时不要调用这个上下文管理器。

## 约定与错误处理

- M4 返回一个 draft 级 verdict；M5 一次批量核验所有 claims，每条必须有且只有一个 verdict。批量方式降低调用数，实际 calls/tokens 交 E 如实记录。
- 支持多句联合证据。每条引用必须在该 target 的检索结果中；证据的原始 `(title, sentence_id, text)` 必须与上下文一致。标题含 `::` 也不会被拆错。
- `SUPPORTED` / `CONTRADICTED` 必须附有效引用；`UNSUPPORTED` 表示证据不足。全部没有证据时直接产生 UNSUPPORTED，不调用模型。
- 空 claims 返回空反馈。空反馈、全部支持、或只有无修改建议的 UNSUPPORTED 反馈直接保留初稿。明确提出合理限定/删除建议的 UNSUPPORTED 可进入同一个改写流程。
- JSON 必须是对象，不能带 Markdown 代码围栏。非法标签、丢失/重复 target、错误引用和不完整编辑记录均抛异常，不能伪装成空成功。
- 核验格式或输入问题抛 `VerificationError`；改写格式或输入问题抛 `RevisionError`；配置/共享类型问题抛 `ConfigurationError`。共享包未安装会明确抛导入错误。C 的调用异常原样传播。E 应保存为 `status="error"`，保留失败分母。
- 提示词只读取合同列出的字段，不传入标准答案或支持事实标签。核验不接收未检索的完整上下文；共用改写接收 benchmark context，便于复核 M3 的批评。

## 修改追踪和日志

返回的 RevisionResult 仍只有 `draft_id`、`feedback_id`、`final_answer`、`final_explanation`。模型原始 JSON 另含 `edits`，每个改变的字段都必须给出 `feedback_target_ids` 和理由，D 在返回前验证它们。

详细 edits 保存在 C/E 的原始响应日志中；E 同时保存原稿、反馈和最终结果即可追溯。D 没有另建日志服务或改变共享 schema。若 C 只保存 token 用量、不保存原始响应，则无法回看详细修改理由；接入时应按合同开启原始输出记录。

## D 的研究分析

`research/D_revision_safety.ipynb` 使用 E 提供的独立正确性标签，统计错→对、对→错、错→错、对→对，并保留失败/缺少标签计数。D 可在表中补充证据缺失、核验误判、忽略反馈、过度修改等案例解释。不要将本实现的核验标签直接当成独立评测标签。

## 当前验证范围

离线测试验证接口、数据边界、引用和接线约束。真实共享 types、C 的实际模型服务、模型的纠错能力、多句推理质量仍需团队集成后验证。完整研究数据尚未运行；demo 和分析模板不能作为实证结果。

目标是 Python 3.11+，不使用第三方库。具体本机验证结果见 `VALIDATION.md`。
