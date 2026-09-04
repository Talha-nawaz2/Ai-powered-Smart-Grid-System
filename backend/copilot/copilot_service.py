"""
AI Grid Copilot Service.

Two modes:
  - LLM mode: real OpenAI tool-calling, used when OPENAI_API_KEY is set.
  - Rule-based fallback: keyword intent routing to the SAME tool functions.
    Used automatically when no API key is configured (or if the LLM call
    fails), so the platform is always demoable.

Concept: notice both modes end up calling the exact same TOOL_REGISTRY
functions. The only thing that differs is WHO decides which tool to call —
an LLM reading natural language, or a regex/keyword router. This is a good
way to internalize what "agentic AI" really is: an LLM given the power to
choose and call functions, nothing more mystical than that.
"""

import re
import json
from sqlalchemy.orm import Session

from backend.utils.config import settings
from backend.utils.logger import logger
from backend.copilot.tools import TOOL_REGISTRY, TOOL_SCHEMAS

SYSTEM_PROMPT = """You are the Grid Copilot, an AI cassistant embedded in a Smart Grid \
Intelligence Platform. You help grid operators understand demand forecasts, transformer \
health, theft risk, and grid topology. Always call a tool to get real data before answering \
factual questions — never guess numbers. Keep answers concise and operational, like a \
control-room assistant, not a chatbot."""


class CopilotService:
    def answer(self, query: str, db: Session) -> dict:
        if settings.OPENAI_API_KEY:
            try:
                return self._answer_with_llm(query, db)
            except Exception as e:
                logger.warning(f"LLM copilot call failed ({e}); falling back to rule-based mode")
        return self._answer_with_rules(query, db)

    # ---------- LLM mode ----------
    def _answer_with_llm(self, query: str, db: Session) -> dict:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(
            model=settings.COPILOT_MODEL, messages=messages,
            tools=TOOL_SCHEMAS, tool_choice="auto",
        )
        msg = response.choices[0].message
        tool_calls_made = []

        if msg.tool_calls:
            messages.append(msg)
            for call in msg.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                fn = TOOL_REGISTRY.get(fn_name)
                result = fn(db, **args) if fn else {"error": f"Unknown tool {fn_name}"}
                tool_calls_made.append({"tool": fn_name, "args": args})
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                })
            final = client.chat.completions.create(model=settings.COPILOT_MODEL, messages=messages)
            answer_text = final.choices[0].message.content
        else:
            answer_text = msg.content

        return {"mode": "llm", "tool_calls": tool_calls_made, "answer": answer_text}

    # ---------- Rule-based fallback ----------
    def _answer_with_rules(self, query: str, db: Session) -> dict:
        q = query.lower()

        tool_calls_made = []

        node_match = re.search(r"(CUS|MTR|TRF)-\d+", query.upper())
        if node_match:
            node_id = node_match.group(0)
            result = TOOL_REGISTRY["trace_consumer_supply"](db, node_id=node_id)
            tool_calls_made.append({"tool": "trace_consumer_supply", "args": {"node_id": node_id}})
            if "error" in result:
                answer = f"Couldn't trace {node_id}: {result['error']}"
            else:
                chain = " -> ".join(f"{n['node_id']} ({n['type']})" for n in result["upstream_path"])
                answer = f"Supply chain for {node_id}: {chain}"

        elif any(w in q for w in ["critical", "substation", "cascade", "single point"]):
            result = TOOL_REGISTRY["get_critical_substations"](db, top_n=5)
            tool_calls_made.append({"tool": "get_critical_substations", "args": {"top_n": 5}})
            if "error" in result:
                answer = result["error"]
            else:
                lines = [f"{s['substation_id']} ({s['connected_transformers']} transformers)"
                         for s in result["critical_substations"]]
                answer = "Most critical substations: " + ", ".join(lines)

        elif any(w in q for w in ["risky", "risk", "transformer fail", "maintenance"]):
            result = TOOL_REGISTRY["get_risky_transformers"](db, threshold=0.8)
            tool_calls_made.append({"tool": "get_risky_transformers", "args": {"threshold": 0.8}})
            if "error" in result:
                answer = result["error"]
            elif result["top_5"]:
                lines = [f"{t['transformer_id']} ({t['failure_probability']:.0%} risk)" for t in result["top_5"]]
                answer = f"{result['count']} transformers exceed 80% failure risk. Top ones: " + ", ".join(lines)
            else:
                answer = "No transformers currently exceed the 80% failure risk threshold."

        elif any(w in q for w in ["theft", "steal", "anomal", "fraud"]):
            result = TOOL_REGISTRY["get_theft_anomalies"](db, limit=5)
            tool_calls_made.append({"tool": "get_theft_anomalies", "args": {"limit": 5}})
            if "error" in result:
                answer = result["error"]
            else:
                lines = [f"{m['meter_id']} (score {m['anomaly_score']})" for m in result["top_anomalies"]]
                answer = "Top suspected theft meters: " + ", ".join(lines)

        elif any(w in q for w in ["forecast", "demand", "tomorrow", "predict"]):
            result = TOOL_REGISTRY["forecast_demand"](db, hours=24)
            tool_calls_made.append({"tool": "forecast_demand", "args": {"hours": 24}})
            if "error" in result:
                answer = result["error"]
            else:
                peak = result["peak"]
                answer = (f"Over the next 24h, peak demand is forecast at "
                          f"{peak['predicted_demand_mw']} MW around {peak['timestamp']}.")

        else:
            result = TOOL_REGISTRY["get_grid_overview"](db)
            tool_calls_made.append({"tool": "get_grid_overview", "args": {}})
            answer = (f"Grid snapshot: {result['total_substations']} substations, "
                      f"{result['total_transformers']} transformers "
                      f"({result['risky_transformer_count']} risky), "
                      f"{result['total_consumers']} consumers, "
                      f"latest demand {result['latest_demand_mw']} MW.")

        return {"mode": "rule-based-fallback", "tool_calls": tool_calls_made, "answer": answer}


copilot_service = CopilotService()
