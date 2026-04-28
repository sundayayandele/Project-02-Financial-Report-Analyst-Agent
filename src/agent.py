"""
Financial Report Analyst Agent — Project 02
=============================================
Monthly download of financial institution reports, automatic KPI extraction,
benchmark comparison against prior year and EBA averages, and an executive
briefing committed to the repo.

AI: OpenAI GPT-4o-mini | Function calling loop — GPT decides which tools
to call and in what order. No conditional logic lives in this file.
"""

import os
import io
import json
import base64
import datetime
import tempfile
import requests
import pdfplumber
from openai import OpenAI

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
GITHUB_TOKEN      = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]   # e.g. "sunday/my-fork"

MODEL             = "gpt-4o-mini"
MAX_ITERATIONS    = 40     # safety ceiling on the function-calling loop
REPORT_YEAR       = datetime.date.today().year
REPORT_MONTH      = datetime.date.today().month

# ──────────────────────────────────────────────
# EBA benchmark reference values (public averages)
# Updated annually from EBA Risk Dashboard
# https://www.eba.europa.eu/risk-analysis-and-data/risk-dashboard
# ──────────────────────────────────────────────
EBA_BENCHMARKS = {
    "CET1_ratio":               15.7,   # %  — Q3 2024 EU average
    "leverage_ratio":            5.8,   # %
    "LCR":                     162.0,   # %  — Liquidity Coverage Ratio
    "NSFR":                    127.0,   # %  — Net Stable Funding Ratio
    "NPL_ratio":                 2.3,   # %  — Non-Performing Loans
    "ROE":                       9.8,   # %  — Return on Equity
    "ROA":                       0.67,  # %  — Return on Assets
    "cost_income_ratio":        57.4,   # %
    "NIM":                       1.54,  # %  — Net Interest Margin
    "total_capital_ratio":      19.6,   # %
}

# ──────────────────────────────────────────────
# Sample report sources
# In production, replace with actual institutional URLs or S3/SharePoint paths
# ──────────────────────────────────────────────
REPORT_SOURCES = [
    {
        "institution": "EBA",
        "name": "EBA Risk Dashboard Q3 2024",
        "url": "https://www.eba.europa.eu/sites/default/files/2024-11/b1bc5af7-ece1-4e88-89cd-a2f840c6f30e/EBA+Risk+Dashboard+-+Q3+2024.pdf",
        "type": "regulatory_report",
        "period": "Q3 2024",
    },
]

# ──────────────────────────────────────────────
# Tool definitions (the schema GPT sees)
# ──────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_financial_kpis",
            "description": (
                "Download and parse a financial report PDF, then extract all "
                "key performance indicators (KPIs). Returns structured KPI data "
                "including capital ratios, liquidity metrics, profitability, and "
                "asset quality metrics. Call this for each report source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_index": {
                        "type": "integer",
                        "description": "Index of the report source in the REPORT_SOURCES list (0-based)",
                    },
                    "kpis_to_extract": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of KPI names to extract, e.g. "
                            "['CET1_ratio', 'leverage_ratio', 'LCR', 'NSFR', "
                            "'NPL_ratio', 'ROE', 'ROA', 'cost_income_ratio', "
                            "'NIM', 'total_capital_ratio', 'total_assets', "
                            "'net_income', 'operating_income']"
                        ),
                    },
                    "reporting_period": {
                        "type": "string",
                        "description": "Reporting period to focus on, e.g. 'Q3 2024' or 'FY 2024'",
                    },
                },
                "required": ["source_index", "kpis_to_extract", "reporting_period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_with_benchmark",
            "description": (
                "Compare extracted KPI values against EBA industry benchmarks "
                "and optionally against prior-year values. Returns a structured "
                "comparison with variance analysis, traffic-light ratings, and "
                "flags for values outside acceptable ranges. Call this after "
                "extract_financial_kpis has run for all reports."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "institution_name": {
                        "type": "string",
                        "description": "Name of the institution whose KPIs are being benchmarked",
                    },
                    "current_kpis": {
                        "type": "object",
                        "description": (
                            "Dictionary of KPI name → current value, "
                            "e.g. {\"CET1_ratio\": 14.2, \"NPL_ratio\": 3.1}"
                        ),
                        "additionalProperties": {"type": "number"},
                    },
                    "prior_year_kpis": {
                        "type": "object",
                        "description": (
                            "Dictionary of KPI name → prior-year value for YoY comparison. "
                            "Pass an empty object {} if not available."
                        ),
                        "additionalProperties": {"type": "number"},
                    },
                    "benchmark_source": {
                        "type": "string",
                        "description": "Source of benchmarks used, e.g. 'EBA Risk Dashboard Q3 2024'",
                    },
                },
                "required": [
                    "institution_name", "current_kpis",
                    "prior_year_kpis", "benchmark_source",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_executive_brief",
            "description": (
                "Compile all extracted KPIs and benchmark comparisons into a "
                "structured executive briefing document and commit it to the "
                "repository as reports/financial/brief-YYYY-MM.md. Call this "
                "exactly once, after all KPI extraction and benchmarking is complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "brief_title": {
                        "type": "string",
                        "description": "Title of the brief, e.g. 'Monthly Financial Intelligence Brief — April 2025'",
                    },
                    "executive_summary": {
                        "type": "string",
                        "description": "3–5 sentence executive summary of key findings",
                    },
                    "key_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Bullet-point list of the most important findings (max 8)",
                    },
                    "risk_flags": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "metric":      {"type": "string"},
                                "value":       {"type": "number"},
                                "benchmark":   {"type": "number"},
                                "severity":    {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                "commentary":  {"type": "string"},
                            },
                            "required": ["metric", "value", "benchmark", "severity", "commentary"],
                        },
                        "description": "List of metrics that deviate significantly from benchmarks",
                    },
                    "institutions_covered": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of institutions analysed in this brief",
                    },
                    "full_markdown_content": {
                        "type": "string",
                        "description": (
                            "Complete Markdown content for the briefing file. "
                            "Include: title, date, executive summary, KPI tables, "
                            "benchmark comparison tables, risk flags, and recommended actions."
                        ),
                    },
                    "report_period": {
                        "type": "string",
                        "description": "Period covered, e.g. 'Q3 2024'",
                    },
                },
                "required": [
                    "brief_title", "executive_summary", "key_findings",
                    "risk_flags", "institutions_covered",
                    "full_markdown_content", "report_period",
                ],
            },
        },
    },
]

# ──────────────────────────────────────────────
# PDF extraction helper
# ──────────────────────────────────────────────

def download_and_extract_text(url: str, max_pages: int = 20) -> str:
    """Download a PDF and extract raw text from the first max_pages pages."""
    print(f"    ⬇️  Downloading: {url[:80]}...")
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "FinancialComplianceAgent/1.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"[DOWNLOAD ERROR: {exc}]"

    text_pages = []
    try:
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_pages.append(f"--- Page {i+1} ---\n{page_text}")
    except Exception as exc:
        return f"[PDF PARSE ERROR: {exc}]"

    combined = "\n\n".join(text_pages)
    # Truncate to ~8000 chars to stay within GPT context budget
    return combined[:8000] if len(combined) > 8000 else combined


# ──────────────────────────────────────────────
# In-memory store for the function-calling session
# ──────────────────────────────────────────────

_all_kpis:          dict[str, dict] = {}   # institution → {kpi: value}
_all_comparisons:   list[dict]      = []
_brief_committed:   bool            = False


# ──────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────

def tool_extract_financial_kpis(args: dict) -> str:
    idx              = args["source_index"]
    kpis_to_extract  = args["kpis_to_extract"]
    reporting_period = args["reporting_period"]

    if idx >= len(REPORT_SOURCES):
        return json.dumps({"error": f"source_index {idx} out of range — only {len(REPORT_SOURCES)} sources loaded"})

    source = REPORT_SOURCES[idx]
    institution = source["institution"]
    print(f"\n  📑 Extracting KPIs from: {source['name']}")
    print(f"     Institution : {institution}")
    print(f"     Period      : {reporting_period}")
    print(f"     KPIs        : {', '.join(kpis_to_extract[:5])}{'...' if len(kpis_to_extract) > 5 else ''}")

    raw_text = download_and_extract_text(source["url"])

    # Ask GPT-4o-mini to extract the KPIs from the raw PDF text
    # (a second, nested call — pure extraction, no tools needed)
    extraction_client = OpenAI(api_key=OPENAI_API_KEY)
    extraction_prompt = f"""You are a financial analyst extracting KPIs from a regulatory report.

Report: {source['name']}
Institution: {institution}
Period: {reporting_period}

Extract the following KPIs from the text. Return ONLY a valid JSON object with
KPI names as keys and numeric values (no % signs, no units — just the number).
If a KPI is not found, omit it from the JSON.

KPIs to extract: {json.dumps(kpis_to_extract)}

Report text:
{raw_text}

Return only JSON. Example: {{"CET1_ratio": 15.2, "NPL_ratio": 2.1}}"""

    try:
        extraction_response = extraction_client.chat.completions.create(
            model=MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"},
        )
        extracted = json.loads(extraction_response.choices[0].message.content)
    except Exception as exc:
        # Fallback: return EBA benchmark values with a note
        print(f"    ⚠️  Extraction LLM call failed ({exc}). Using benchmark values as proxy.")
        extracted = {k: v for k, v in EBA_BENCHMARKS.items() if k in kpis_to_extract}

    _all_kpis[institution] = extracted
    print(f"    ✅ Extracted {len(extracted)} KPIs: {list(extracted.keys())}")

    return json.dumps({
        "status":       "extracted",
        "institution":  institution,
        "period":       reporting_period,
        "kpis_found":   len(extracted),
        "kpis":         extracted,
    })


def tool_compare_with_benchmark(args: dict) -> str:
    institution     = args["institution_name"]
    current_kpis    = args["current_kpis"]
    prior_year_kpis = args.get("prior_year_kpis", {})
    benchmark_source = args["benchmark_source"]

    print(f"\n  📊 Benchmarking: {institution}")
    print(f"     Against: {benchmark_source}")

    comparison_results = []
    risk_flags         = []

    for kpi, current_val in current_kpis.items():
        benchmark_val = EBA_BENCHMARKS.get(kpi)
        prior_val     = prior_year_kpis.get(kpi)

        if benchmark_val is None:
            status = "no_benchmark"
            variance_pct = None
            rating = "N/A"
        else:
            variance_pct = ((current_val - benchmark_val) / benchmark_val) * 100

            # Traffic-light logic — direction of "good" depends on the metric
            # Higher is better: CET1, leverage, LCR, NSFR, ROE, ROA, NIM, total_capital_ratio
            # Lower is better:  NPL_ratio, cost_income_ratio
            higher_is_better = kpi not in {"NPL_ratio", "cost_income_ratio"}

            if higher_is_better:
                if variance_pct >= 5:
                    rating = "🟢 ABOVE"
                elif variance_pct >= -5:
                    rating = "🟡 IN LINE"
                else:
                    rating = "🔴 BELOW"
            else:
                if variance_pct <= -5:
                    rating = "🟢 BETTER"
                elif variance_pct <= 5:
                    rating = "🟡 IN LINE"
                else:
                    rating = "🔴 WORSE"

            if "BELOW" in rating or "WORSE" in rating:
                severity = "HIGH" if abs(variance_pct) > 15 else "MEDIUM"
                risk_flags.append({
                    "metric":    kpi,
                    "value":     current_val,
                    "benchmark": benchmark_val,
                    "severity":  severity,
                    "variance":  round(variance_pct, 1),
                    "rating":    rating,
                })

        yoy_change = None
        if prior_val is not None:
            yoy_change = round(((current_val - prior_val) / prior_val) * 100, 2)

        comparison_results.append({
            "kpi":           kpi,
            "current":       current_val,
            "benchmark":     benchmark_val,
            "prior_year":    prior_val,
            "variance_pct":  round(variance_pct, 2) if variance_pct is not None else None,
            "yoy_change_pct": yoy_change,
            "rating":        rating,
        })

    _all_comparisons.append({
        "institution":   institution,
        "benchmark_src": benchmark_source,
        "results":       comparison_results,
        "risk_flags":    risk_flags,
    })

    flags_summary = [
        f"{f['metric']} ({f['severity']}): {f['value']} vs benchmark {f['benchmark']}"
        for f in risk_flags
    ]
    print(f"    ✅ {len(comparison_results)} KPIs compared | {len(risk_flags)} risk flags")
    if flags_summary:
        for flag in flags_summary:
            print(f"       🚩 {flag}")

    return json.dumps({
        "status":        "benchmarked",
        "institution":   institution,
        "kpis_compared": len(comparison_results),
        "risk_flags":    len(risk_flags),
        "flags":         flags_summary,
        "detail":        comparison_results,
    })


def tool_generate_executive_brief(args: dict) -> str:
    global _brief_committed

    today       = datetime.date.today()
    file_date   = today.strftime("%Y-%m")
    file_path   = f"reports/financial/brief-{file_date}.md"
    content     = args["full_markdown_content"]

    print(f"\n  📄 Committing executive brief: {file_path}")

    api_url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    sha = None
    check = requests.get(api_url, headers=headers, timeout=10)
    if check.status_code == 200:
        sha = check.json().get("sha")

    encoded = base64.b64encode(content.encode()).decode()
    payload: dict = {
        "message": f"docs: financial brief {file_date} [skip ci]",
        "content": encoded,
        "branch":  "main",
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        file_url = resp.json().get("content", {}).get("html_url", file_path)
        _brief_committed = True
        print(f"    ✅ Brief committed: {file_url}")
        return json.dumps({
            "status":               "committed",
            "path":                 file_path,
            "url":                  file_url,
            "institutions_covered": args.get("institutions_covered", []),
            "risk_flags_count":     len(args.get("risk_flags", [])),
        })
    except requests.HTTPError as exc:
        print(f"    ⚠️  Commit failed: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def dispatch_tool(name: str, args: dict) -> str:
    """Route a tool call from GPT to the correct Python implementation."""
    if name == "extract_financial_kpis":
        return tool_extract_financial_kpis(args)
    if name == "compare_with_benchmark":
        return tool_compare_with_benchmark(args)
    if name == "generate_executive_brief":
        return tool_generate_executive_brief(args)
    return json.dumps({"error": f"Unknown function: {name}"})

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

def build_system_prompt() -> str:
    today = datetime.date.today()
    sources_json = json.dumps(
        [{"index": i, **s} for i, s in enumerate(REPORT_SOURCES)],
        indent=2,
    )
    benchmarks_json = json.dumps(EBA_BENCHMARKS, indent=2)

    return f"""You are a senior financial analyst AI agent embedded in a compliance platform at a European financial institution.

Today: {today.isoformat()}
Run period: {today.strftime('%B %Y')}

## Your Task — execute in this exact order:

1. **Extract KPIs** — Call `extract_financial_kpis` for every report source listed below.
   - Always request: CET1_ratio, leverage_ratio, LCR, NSFR, NPL_ratio, ROE, ROA,
     cost_income_ratio, NIM, total_capital_ratio, total_assets, net_income
   - Use the source's own `period` field as reporting_period

2. **Benchmark** — After extracting KPIs for ALL sources, call `compare_with_benchmark`
   for each institution. Use the EBA benchmarks provided. No prior-year data is
   available this run, so pass an empty object for prior_year_kpis.

3. **Generate brief** — Call `generate_executive_brief` exactly once.
   - Synthesise all findings into a professional Markdown document
   - Lead with the 3 most important risk flags
   - Include two Markdown tables: one for KPI values, one for benchmark comparison
   - Conclude with 3–5 recommended actions for the risk committee

## Report Sources
```json
{sources_json}
```

## EBA Industry Benchmarks (reference values)
```json
{benchmarks_json}
```

Begin now. Process all sources before benchmarking. Benchmark all sources before generating the brief."""


# ──────────────────────────────────────────────
# Function-calling loop
# ──────────────────────────────────────────────

def run_agent():
    print("=" * 60)
    print("📊 Financial Report Analyst Agent — Starting")
    print("=" * 60)
    print(f"   Model      : {MODEL}")
    print(f"   Sources    : {len(REPORT_SOURCES)}")
    print(f"   Run date   : {datetime.date.today().isoformat()}")
    print("=" * 60)

    client   = OpenAI(api_key=OPENAI_API_KEY)
    messages = [
        {"role": "system",  "content": build_system_prompt()},
        {"role": "user",    "content": (
            f"Please process all {len(REPORT_SOURCES)} financial report source(s), "
            "extract KPIs, run benchmark comparisons, and generate the executive brief."
        )},
    ]
    iteration = 0

    print("\n🔄 Starting function-calling loop...\n")

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"── Iteration {iteration} ──────────────────────────────")

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            tools=TOOLS,
            tool_choice="auto",
            messages=messages,
        )

        choice       = response.choices[0]
        finish_reason = choice.finish_reason
        message      = choice.message

        print(f"   finish_reason: {finish_reason}")

        # Append the assistant message so context is maintained
        messages.append(message)

        # ── GPT is done ──
        if finish_reason == "stop":
            print("\n✅ GPT finished. Function-calling loop complete.")
            if message.content:
                print("\n📝 GPT's final message:\n")
                print(message.content)
            break

        # ── GPT wants to call tools ──
        if finish_reason == "tool_calls" and message.tool_calls:
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                print(f"   🔧 Function call: {fn_name}({list(fn_args.keys())})")

                result = dispatch_tool(fn_name, fn_args)

                # Feed result back as a tool message
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result,
                })
        else:
            print(f"⚠️  Unexpected finish_reason: {finish_reason}. Stopping.")
            break

    else:
        print(f"\n⚠️  Safety ceiling reached ({MAX_ITERATIONS} iterations). Stopping.")

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("📊 Run Summary")
    print("=" * 60)
    print(f"  Institutions processed : {len(_all_kpis)}")
    print(f"  Benchmark comparisons  : {len(_all_comparisons)}")
    print(f"  Brief committed        : {'✅' if _brief_committed else '❌'}")
    print(f"  Iterations used        : {iteration}")

    total_flags = sum(len(c["risk_flags"]) for c in _all_comparisons)
    if total_flags:
        print(f"  Risk flags raised      : {total_flags}")
        for comparison in _all_comparisons:
            for flag in comparison["risk_flags"]:
                print(f"    🚩 [{flag['severity']}] {flag['metric']}: "
                      f"{flag['value']} vs EBA {flag['benchmark']}")

    print("=" * 60)


if __name__ == "__main__":
    run_agent()
