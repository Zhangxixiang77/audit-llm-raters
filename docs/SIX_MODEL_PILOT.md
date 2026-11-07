# 六模型真实试跑方案

核对日期：2026-11-07。本文只记录非敏感配置、预算和执行门槛，不包含作文、密钥、原始响应或评分结果。

## 1. 冻结的候选组合

| 内部名称 | 调用模型／快照 | 运行方式 | 思考控制 | 标准输入／输出价（每百万 token） | 当前状态 |
|---|---|---|---|---|---|
| `minimax_m3` | `MiniMax-M3` | MiniMax API | `thinking.type=disabled`；不发送 temperature | ¥2.10／¥8.40 | 已完成 400 条，原库复用 |
| `openai_gpt_6_1_terra` | `gpt-6.1-terra` | OpenAI Chat Completions | `reasoning_effort=none`；不发送 temperature | $1.00／$5.00 | 待凭据和单篇冒烟 |
| `anthropic_claude_haiku_4_5` | `claude-haiku-4-5-20251001` | Anthropic Messages | `thinking.type=disabled` | $1.00／$5.00 | 待凭据和单篇冒烟 |
| `deepseek_v4_1_flash` | `deepseek-flash`，文档对应 DeepSeek-V4.1-Flash-1017 | DeepSeek Chat Completions | `thinking.type=disabled` | 峰值、缓存未命中 $0.30／$1.20 | 待凭据和单篇冒烟 |
| `qwen_3_6_plus` | `qwen3.6-plus-2026-04-02` | 百炼 OpenAI 兼容接口 | `enable_thinking=false` | ¥2.00／¥12.00 | 待凭据和单篇冒烟 |
| `hf_llama_3_1_8b_instruct` | `meta-llama/Llama-3.1-8B-Instruct`，执行前固定完整 commit | 外部 GPU 上的 vLLM，OpenAI 兼容接口 | 非混合思考模型；temperature=0 | 按 GPU 时间计费，不能伪造 token 单价 | 等待外部 GPU、模型授权和 commit |

选择 Qwen 而不是当前 Gemini，是为了保持“关闭思考”的实验口径。当前 Gemini 3.5 Flash-Lite 的 `MINIMAL` 只是最低思考档，并不保证完全关闭；Gemini 2.5 又限制为历史项目使用。此处不静默改变研究条件。

OpenAI 和 DeepSeek 的当前 API 标识不是完整的日期快照。执行时保留完整原始响应，并记录响应中的模型标识、`system_fingerprint`（如有）、配置哈希和调用日期；不得把别名解析成未经验证的其他版本。

## 2. 为什么不直接在本机运行 Llama

当前执行环境没有可用 NVIDIA GPU，只有约 2 个 CPU 核和 4 GiB 内存，不能诚实地声称可以本地加载 8B 级 BF16 模型。开放权重路线需要一个外部 vLLM 端点；工程上建议使用约 24 GB 显存的 GPU，并先确认模型授权、完整 Hugging Face commit、服务日志和实际显存占用。量化版本不能未经讨论直接替代 BF16 版本。

GPU 费用按小时和最长运行时间设置外部硬上限。API 的 token 费用上限不能约束 GPU 计费，因此两部分必须分别审批和记录。

## 3. 预算预览

下表已经用真实 prompt 做了命令行预览：输入按字符数约 4 字符/token 估算，输出按 200 token 上限估算，不使用缓存、Batch 或非高峰折扣。它是预算基准，不是最终账单。作为交叉核对，MiniMax 已完成的同一批 400 次调用实际使用 858,442 输入 token、22,313 输出 token；不同模型的 tokenizer 和输出长度会不同。

| 模型 | 400 次调用预览 | 单篇 8 次冒烟预览 | 批量硬上限 |
|---|---:|---:|---:|
| MiniMax-M3 | 已实际支出 ¥1.9901574；不重新运行 | 不需要 | 不新增 |
| GPT-6.1 Terra | $1.387084 | $0.035080 | $2.00 |
| Claude Haiku 4.5 | $1.387084 | $0.035080 | $2.00 |
| DeepSeek-V4.1 Flash | $0.392125（按峰值保守价） | $0.010044 | $0.60 |
| Qwen 3.6 Plus | ¥2.934168 | ¥0.073360 | ¥4.00 |
| Llama 3.1 8B | 外部 GPU 报价确定后计算 | 先测 8 次和启动时间 | 单独设置 GPU 小时／金额上限 |

四个新增 API 模型的预览合计为 **$3.166293 + ¥2.934168**。对应批量硬上限合计为 **$4.60 + ¥4.00**。不同币种不得直接相加。冒烟建议上限分别为：OpenAI $0.05、Claude $0.05、DeepSeek $0.02、Qwen ¥0.10。

实际执行前必须重新预览；若预览或账户区域价格与本文不同，先更新预算，不直接运行。

## 4. 单篇冒烟门槛

对同一篇 pilot 作文运行完整的 `1 模型 × 2 提示 × 2 细则 × 2 重复 = 8` 次调用。先不带 `--execute` 预览，再设置对应币种的 `--max-cost` 执行。

全部满足才允许进入 50 篇批量：

1. 8 个确定性 request ID 唯一，原始响应完整落盘且非空。
2. 8 条均为 `parse_status=ok`，分数为 1–5 的整数，reason 非空。
3. 没有 length／max-token 截断；输入、输出 token usage 完整。
4. 原始响应与请求参数一致，没有意外思考输出；供应商提供 reasoning token 计数时应为 0。
5. 返回模型标识没有发生未经记录的替换，实际费用没有超过冒烟上限。
6. 任一条件失败就停止该模型，保留失败分型，不自动换模型、不重写成成功、不进入批量。

每个模型使用独立 SQLite 数据库，既避免跨币种求和，也避免误重复调用已经完成的 MiniMax。

## 5. 批量和合并验收

其余五个模型各运行 400 条，共新增 2,000 条；合并既有 MiniMax 400 条后应为：

`50 essays × 6 models × 2 prompts × 2 rubrics × 2 repeats = 2,400 records`

合并前逐库检查 request ID、模型版本、原始响应、解析分型、token 和费用。合并后检查 2,400 个 request ID 唯一、每篇每模型 8 条、50 篇作文与既有 pilot 完全一致、held-out anchors 没有进入评分集。通过后才运行固定顺序的 QC → G/D → MFRM → 压缩 → DRF → 稳健性分析。ELLIPSE 全部为 ELL，组间 DRF 仍应标为不可估计。

## 6. 本地凭据名称

只从环境变量读取，不写入 YAML、数据库配置、报告或 GitHub：

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`
- 可选的远程 vLLM 凭据：`VLLM_API_KEY`

公开配置见 `configs/models.six-model.example.yaml`；真实的单模型本地配置使用 `configs/local*.yaml`，已被 Git 忽略。

## 7. 官方核对来源

- OpenAI 模型页：https://developers.openai.com/api/docs/models/gpt-6.1-terra
- Anthropic 模型与思考说明：https://platform.claude.com/docs/en/claude_api_primer
- Anthropic 价格：https://platform.claude.com/docs/en/about-claude/pricing
- DeepSeek 模型与价格：https://api-docs.deepseek.com/quick_start/pricing/
- Qwen 深度思考：https://help.aliyun.com/zh/model-studio/deep-thinking
- Qwen 价格：https://help.aliyun.com/zh/model-studio/model-pricing
- Qwen 3.6 Plus 快照：https://help.aliyun.com/zh/model-studio/qwen3-6-plus
- Hugging Face 模型：https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
