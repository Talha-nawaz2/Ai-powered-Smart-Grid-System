"""
Grid Knowledge Graph Service (NetworkX).

Concept: We model the grid as a DIRECTED graph, edges pointing "upstream"
(Consumer -> Meter -> Transformer -> Substation -> Power Plant), plus
undirected-in-spirit Substation<->Substation transmission line edges
(added both directions). Direction matters here: it turns "what powers
this consumer?" into a simple walk along successors, instead of an
ambiguous search on an undirected graph.

Simplification note: there's no explicit substation->plant table in our
data, so we connect each substation to power plants in the SAME REGION
as a reasonable approximation of real grid feeding patterns. This is
exactly the kind of assumption you'd flag and revisit with a real utility
partner in a production system — good to be upfront about it, not hide it.
"""

import pickle
import networkx as nx

from backend.utils.config import settings
from backend.utils.logger import logger

GRAPH_PATH = settings.DATA_DIR / "models" / "grid_graph.pkl"


class GridGraphService:
    def __init__(self):
        self.graph: nx.DiGraph | None = None

    def build(self, repo) -> dict:
        G = nx.DiGraph()

        plants = repo.get_power_plants()
        substations = repo.get_substations(limit=5000)
        transformers = repo.get_transformers(limit=5000)
        meters = repo.get_meters(limit=5000)
        consumers = repo.get_consumers(limit=5000)
        lines = repo.get_transmission_lines()

        for p in plants:
            G.add_node(p["plant_id"], type="power_plant", region=p["region"],
                       capacity_mw=p["capacity_mw"])

        plants_by_region: dict[str, list[str]] = {}
        for p in plants:
            plants_by_region.setdefault(p["region"], []).append(p["plant_id"])

        for s in substations:
            G.add_node(s["substation_id"], type="substation", region=s["region"],
                       capacity_mva=s["capacity_mva"])
            # Simplification: connect to up to 2 plants in the same region
            for pid in plants_by_region.get(s["region"], [])[:2]:
                G.add_edge(s["substation_id"], pid, relation="fed_by")

        for t in transformers:
            G.add_node(t["transformer_id"], type="transformer", health_score=t["health_score"])
            G.add_edge(t["transformer_id"], t["substation_id"], relation="connected_to")

        for m in meters:
            G.add_node(m["meter_id"], type="meter")
            G.add_edge(m["meter_id"], m["transformer_id"], relation="connected_to")
            G.add_edge(m["consumer_id"], m["meter_id"], relation="served_by")

        for c in consumers:
            G.add_node(c["consumer_id"], type="consumer", region=c["region"],
                       consumer_type=c["consumer_type"])

        for line in lines:
            # transmission lines are genuinely bidirectional (power can flow either way)
            G.add_edge(line["from_substation"], line["to_substation"],
                       relation="transmission_line", voltage_kv=line["voltage_kv"])
            G.add_edge(line["to_substation"], line["from_substation"],
                       relation="transmission_line", voltage_kv=line["voltage_kv"])

        self.graph = G
        self.save()

        stats = self.get_stats()
        logger.info(f"Knowledge graph built: {stats}")
        return stats

    def save(self):
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self) -> bool:
        if GRAPH_PATH.exists():
            with open(GRAPH_PATH, "rb") as f:
                self.graph = pickle.load(f)
            return True
        return False

    def _ensure_loaded(self):
        if self.graph is None and not self.load():
            raise RuntimeError("Knowledge graph not built yet. Call /api/graph/build first.")

    def get_stats(self) -> dict:
        self._ensure_loaded()
        G = self.graph
        type_counts: dict[str, int] = {}
        for _, data in G.nodes(data=True):
            type_counts[data.get("type", "unknown")] = type_counts.get(data.get("type", "unknown"), 0) + 1
        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "node_counts_by_type": type_counts,
            "is_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
        }

    def trace_upstream(self, node_id: str) -> list[dict]:
        """Walks from a consumer/meter/transformer all the way up to its power plant(s)."""
        self._ensure_loaded()
        if node_id not in self.graph:
            raise ValueError(f"Node '{node_id}' not found in graph")

        path = [{"node_id": node_id, "type": self.graph.nodes[node_id].get("type")}]
        current = node_id
        visited = {node_id}
        while True:
            successors = [
                s for s in self.graph.successors(current)
                if self.graph.edges[current, s]["relation"] in ("served_by", "connected_to", "fed_by")
                and s not in visited
            ]
            if not successors:
                break
            current = successors[0]
            visited.add(current)
            path.append({"node_id": current, "type": self.graph.nodes[current].get("type")})
        return path

    def shortest_path(self, source: str, target: str) -> dict:
        self._ensure_loaded()
        if source not in self.graph or target not in self.graph:
            raise ValueError("Source or target node not found in graph")
        undirected = self.graph.to_undirected()
        path = nx.shortest_path(undirected, source, target)
        return {
            "source": source, "target": target, "hops": len(path) - 1,
            "path": [{"node_id": n, "type": self.graph.nodes[n].get("type")} for n in path],
        }

    def neighbors(self, node_id: str) -> dict:
        self._ensure_loaded()
        if node_id not in self.graph:
            raise ValueError(f"Node '{node_id}' not found in graph")
        upstream = [{"node_id": s, "relation": self.graph.edges[node_id, s]["relation"]}
                    for s in self.graph.successors(node_id)]
        downstream = [{"node_id": p, "relation": self.graph.edges[p, node_id]["relation"]}
                      for p in self.graph.predecessors(node_id)]
        return {"node_id": node_id, "type": self.graph.nodes[node_id].get("type"),
                "upstream": upstream, "downstream_count": len(downstream), "downstream_sample": downstream[:10]}

    def critical_substations(self, top_n: int = 10) -> list[dict]:
        """
        Substations with the most downstream transformers connected are 'critical' —
        a failure there cascades to the most consumers. Simple in-degree ranking.
        """
        self._ensure_loaded()
        substations = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "substation"]
        ranked = sorted(
            substations,
            key=lambda s: self.graph.in_degree(s),
            reverse=True,
        )[:top_n]
        return [
            {"substation_id": s, "connected_transformers": self.graph.in_degree(s)}
            for s in ranked
        ]


graph_service = GridGraphService()
