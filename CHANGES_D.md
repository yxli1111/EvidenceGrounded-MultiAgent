# D 修复说明 — 2026-10-29

## 短答案格式

`prompts/revision/revise_answer.txt` 现在明确要求 `final_answer` 只包含最短答案片段：完整实体名、地点、数字、日期或小写 yes/no。推理和依据只写在 `final_explanation`。增加正反例，避免把 England 写成 She was born in England。不会按字数硬截断，以免损坏多词实体名。

用户随后补充了 A 的真实实现。已直接修改其正在加载的 `prompts/baselines/self_revise_v1.md`：在 SYSTEM 中要求 short_answer 是完整实体/数字/日期/yes-no 短片段，推理只进 explanation，并增加正反例。无需改 A 的 baselines.py 或字段名称，按同一路径替换提示词即可生效。

A 的 answer.load_prompt 本身按提示词内容计算 SHA256 版本，修改会自动反映到其 prompt_versions；文件名沿用 self_revise_v1.md，以兼容现有 PROMPT_FILES 路径。没有创建第二个未被加载的提示词文件。

## 两种编辑记录错误

旧版要求模型对每个变化字段恰好返回一条 edits，超出了 RevisionResult 的四字段合同，导致本来有效的答案也会失败。

现在以实际的原稿/最终文本差异建立审计，模型 edits 可选。漏记、重复同字段、列出未变化字段、使用 short_answer/explanation 字段别名或无效可选注释，会被归一化/记录警告，而不把有效答案标成失败。自动记录明确标为 computed_diff；没有模型理由时 model_notes 为空，不会编造理由。

输入 feedback、必需答案字段类型、JSON、模型服务异常仍严格处理。没有增加第二次改写调用。M3/M4/M5 继续使用相同改写函数、提示词和配置。

## M5 引用错误

M5 从一次批量核验改为逐条有证据 claim 单独核验，让模型每次只看到当前 claim 的证据。空证据直接返回 UNSUPPORTED，不调用模型。

提供可直接复制的唯一短引用编号，例如 C1E1、C1E2、C2E1，并显式列出 allowed_evidence_ids。程序只接受当前 target 的编号或标准 ID，统一映射回 title::sentence_id；不会重新切分标题或猜测句号。别名和标准 ID 重复指向同一句时去重。

跨 claim / 不存在的编号依然报错，错误信息增加 target、错误引用和允许编号，便于排查。此改动减少串引和手写 ID 错误，但不能保证模型不再输出非法引用。

## 实验设置

- 提示词版本更新为 d-draft-v2、d-claims-v2-per-claim、d-revision-v2，提示词内容也已变化，精确请求缓存会区分新旧输入。
- M5 核验调用数从每题一批变为每条有证据 claim 一次。E 应记录真实调用、tokens、延迟和成本；不要与旧批量运行混在同一设置下汇总。
- 没有修改共享 contracts、B 的检索实现或 E 的评分逻辑。
- 详细审计使用 src.agents.revision logger 的 INFO 输出；C/E 继续保存原始请求、响应、原稿和最终结果。

## 替换与验证

修复压缩包只包含新增/修改文件，按仓库根目录解压合入；保留已有团队 contracts。随后运行：

```powershell
python -m unittest discover -s tests_d -v
python -m examples.demo_d --output demo_result.json
```

真实共享类型回归测试可通过环境变量 D_TEAM_ROOT 指向包含 contracts 的团队仓库；A 实际提示词回归可通过 D_A_ROOT 指向包含 A 代码的仓库。没有可用代码路径时对应集成项会跳过，独立 D mock 测试仍运行。测试不访问网络或真实模型。
