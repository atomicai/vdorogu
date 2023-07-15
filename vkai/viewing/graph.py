from typing import Dict

from networkx.drawing.nx_agraph import to_agraph


def visualize_graph(_G, _meta: Dict = None, filename: str = "graph.png"):
    if _meta:
        nodes = _G.nodes()
        for _node_idx in nodes:
            _node = nodes[_node_idx]
            _label = _node.get("label", None)
            if _label is not None:
                _label_ios = _meta[_label].get("ios", -1)
                _label_droid = _meta[_label].get("droid", -1)
                if _label_droid * _label_ios < 0:
                    _node["fillcolor"] = "pink"
                    if _label_ios < 0:
                        _node["label"] += " DROID " + str(_label_droid)
                    else:
                        _node["label"] += " iOS " + str(_label_droid)
                else:
                    _node["label"] += " iOS " + str(_label_ios) + " DROID " + str(_label_droid)
    A = to_agraph(_G)
    A.layout("neato")
    A.draw(str(filename))
    return A


__all__ = ["visualize_graph"]
