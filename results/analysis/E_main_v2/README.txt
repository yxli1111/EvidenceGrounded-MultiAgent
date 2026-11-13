E 结果包（正式运行 main_v2，100 道 eval 题，gpt-4o-mini，0 次失败）

给 PPT：
  E_results_main_v2.docx        RQ1 总表、验证标签、RQ3 图和表、Conclusions、Limitations（英文，可直接复制）
  table_rq1_results.png         RQ1 总表图片（可直接贴进 slide）
  table_rq3_cost.png            RQ3 成本表图片
  table_verifier_labels.png     M4/M5 验证标签与 gold supporting-fact recall
  rq3_cost_vs_em_main_v2.png    RQ3 成本-准确率散点图

给 RQ2 / 分析（data 文件夹）：
  summary_main_v2.json          所有汇总数字（EM、CI、ΔEM、修改前后对比、成本、延迟、验证标签）
  rq2_m4_vs_m5_per_example.csv  每题 M4 与 M5 并排：EM、F1、成本、延迟、调用次数、证据数、claim 数、各标签数、是否改答案、SF recall
  rq2_m5_claims.csv             M5 每条 claim：文本、验证标签、引用证据、检索到的 gold 句数、理由、草稿/最终/gold 答案
  rq2_m4_verification.csv       M4 每题的验证标签、理由、修改建议、草稿/最终/gold 答案
  revision_flips.csv            M2–M5 所有「改对」和「改错」的题目
  main_v2_records.jsonl         完整原始记录（每题每方法一行：草稿、证据、claims、feedback、最终答案、usage）
  main_v2_per_example.csv       runner 输出的逐题表
  main_v2_cost_summary.json / main_v2_metadata.json / main_v2_gold_lookup.json

口径：EM/F1 为 HotpotQA 官方口径，失败计 0（本次无失败）；CI 为配对 bootstrap 95%（5000 次，seed 0）；
每题成本含共享草稿。M1–M5 与 pilot（main_n100）用同一批 100 题，草稿相同。
