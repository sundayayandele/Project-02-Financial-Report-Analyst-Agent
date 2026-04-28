# Project-02-Financial-Report-Analyst-Agent
# 📊 Project 02 — Financial Report Analyst Agent

> **Category A: GitHub Actions + AI API** — runs entirely in GitHub's cloud. No local setup needed.

An autonomous AI agent that runs on the **1st of every month**, downloads financial institution reports from EBA/ECB/ESMA, extracts KPIs using GPT-4o-mini's function calling, benchmarks them against EBA industry averages, and commits a structured executive briefing to the repo.

---

## 🤖 Why It's Agentic

GPT-4o-mini is given three tools and a goal. It decides:

- Which KPIs to extract from each report
- Whether to raise a risk flag on a given metric
- How to structure the executive brief based on what it found

You write the tools. GPT writes the analysis. That's the function-calling loop.

```
┌─────────────────────────────────────────────────────────┐
│                  FUNCTION-CALLING LOOP                  │
│                                                         │
│  1. GPT receives report sources + EBA benchmarks        │
│  2. GPT calls extract_financial_kpis(source_index=0)    │
│     → agent downloads PDF, extracts text, LLM parses    │
│  3. GPT calls compare_with_benchmark(institution=...)   │
│     → traffic-light vs EBA averages + YoY delta         │
│  4. GPT calls generate_executive_brief(...)             │
│     → Markdown report committed to repo                 │
│  5. GPT returns finish_reason="stop" → loop exits       │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
project-02-financial-report-analyst-agent/
├── README.md
├── requirements.txt
├── src/
│   └── agent.py                        ← All agentic logic
└── .github/
    └── workflows/
        └── financial-analyst.yml       ← Runs 1st of each month 06:00 UTC
```

**Output location (auto-committed each run):**
```
reports/
└── financial/
    └── brief-YYYY-MM.md
```

---

## ⚙️ Step-by-Step Setup

### 1. Fork this repository

### 2. Add your OpenAI API key as a secret

1. Go to **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name: `OPENAI_API_KEY`
4. Value: your key from [platform.openai.com](https://platform.openai.com/api-keys)

`GITHUB_TOKEN` is provided automatically by GitHub Actions — no extra setup.

### 3. Enable GitHub Actions

Actions tab → **"I understand my workflows, enable them"**

### 4. Run manually (first time)

Actions tab → **📊 Financial Report Analyst Agent** → **Run workflow**

---

## 🛠️ Tools Available to GPT

| Tool | When GPT calls it | Output |
|------|-------------------|--------|
| `extract_financial_kpis` | Once per report source | JSON of KPI name → numeric value |
| `compare_with_benchmark` | After all extractions complete | Traffic-light table vs EBA averages |
| `generate_executive_brief` | Once, at the very end | Committed Markdown brief |

---

## 📊 EBA Benchmarks Used (Q3 2024)

| KPI | EBA Average | Direction |
|-----|-------------|-----------|
| CET1 Ratio | 15.7% | Higher = better |
| Leverage Ratio | 5.8% | Higher = better |
| LCR | 162% | Higher = better |
| NSFR | 127% | Higher = better |
| NPL Ratio | 2.3% | Lower = better |
| ROE | 9.8% | Higher = better |
| ROA | 0.67% | Higher = better |
| Cost/Income | 57.4% | Lower = better |
| NIM | 1.54% | Higher = better |
| Total Capital Ratio | 19.6% | Higher = better |

Source: [EBA Risk Dashboard](https://www.eba.europa.eu/risk-analysis-and-data/risk-dashboard)

---

## 📄 Output Example — `reports/financial/brief-2025-04.md`

```markdown
# Monthly Financial Intelligence Brief — April 2025

## Executive Summary
EBA aggregate data for Q3 2024 shows the EU banking sector remains well capitalised
with a CET1 ratio of 15.7%, comfortably above minimum requirements ...

## Key Findings
- ✅ CET1 ratio in line with 12-month trend
- 🚩 NPL ratio trending up in southern European cohort
- ✅ LCR well above 100% minimum across all jurisdictions

## KPI Summary Table
| Metric | Current | EBA Benchmark | Status |
|--------|---------|---------------|--------|
| CET1   | 15.7%   | 15.7%         | 🟡 IN LINE |
...

## Risk Flags
| Metric | Value | Benchmark | Severity | Commentary |
...

## Recommended Actions
1. Escalate NPL trend to credit risk committee ...
```

---

## 🔍 Key Code Concept — The Function-Calling Loop

```python
while iteration < max_iterations:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=TOOLS,
        tool_choice="auto",
        messages=messages,
    )
    choice = response.choices[0]

    if choice.finish_reason == "stop":
        break  # GPT decided it's done

    if choice.finish_reason == "tool_calls":
        for tool_call in choice.message.tool_calls:
            result = dispatch_tool(tool_call.function.name,
                                   json.loads(tool_call.function.arguments))
            # Feed result back — GPT sees it and continues reasoning
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
```

The assistant message must be appended **before** the tool results — OpenAI's API requires the full turn sequence in messages history.

---

## 🔗 Resources

- [OpenAI Function Calling Docs](https://platform.openai.com/docs/guides/function-calling)
- [EBA Risk Dashboard](https://www.eba.europa.eu/risk-analysis-and-data/risk-dashboard)
- [pdfplumber Docs](https://github.com/jsvine/pdfplumber)
- [GitHub Actions Docs](https://docs.github.com/en/actions)

---

## 📄 License

MIT — free to fork, adapt, and deploy in your own financial intelligence workflows.
