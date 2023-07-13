import datetime as dt
import json
from pathlib import Path

import graphviz
import networkx as nx
from networkx.readwrite import json_graph
from tqdm.notebook import tqdm

from .filesystem import REPO_PATH, Filesystem
from .table_in_fs import TableInFilesystem


class DependencyGraph:
    DEFAULT_PATH = REPO_PATH / 'dags' / 'trgetl-dependency-graph-metadata.json'
    REFRESH_DATETIME_NODE = '__refresh_datetime__'

    @property
    def graph(self):
        graph = self._graph
        if graph is not None:
            self._alert_graph_expiration(graph)
        return graph

    @graph.setter
    def graph(self, graph):
        if graph is not None:
            self._alert_graph_expiration(graph)
        self._graph = graph

    def __init__(self, use_cache=True, cache_path=None, refresh_delay=2):
        if cache_path is None:
            cache_path = self.DEFAULT_PATH
        elif isinstance(cache_path, str):
            cache_path = Path(cache_path)

        if isinstance(refresh_delay, (int, float)):
            refresh_delay = dt.timedelta(days=1) * refresh_delay

        self.use_cache = use_cache
        self.cache_path = cache_path
        self.refresh_delay = refresh_delay

        self._graph = None
        if self.use_cache:
            self.graph = self._read_graph_from_cache()
        else:
            self.graph = self._read_graph_from_filesystem()

    def refresh(self):
        graph = self._read_graph_from_filesystem()
        if self.use_cache:
            self._write_graph_to_cache(graph)
        self.graph = graph

    def predecessors(self, table_name, depth_limit=1, tree=False):
        predecessors = self._recursive_successors_or_predesessors(
            table_name,
            depth_left=depth_limit,
            method='predecessors',
        )
        if not tree:
            predecessors = (predecessor for group in predecessors.values() for predecessor in group)
            predecessors = sorted(set(predecessors))
        return predecessors

    def successors(self, table_name, depth_limit=1, tree=False):
        successors = self._recursive_successors_or_predesessors(
            table_name,
            depth_left=depth_limit,
            method='successors',
        )
        if not tree:
            successors = (successor for group in successors.values() for successor in group)
            successors = sorted(set(successors))
        return successors

    def draw_table_links(
        self,
        table_name,
        predecessor_depth_limit=1,
        successors_depth_limit=1,
    ):
        predecessors = self.predecessors(table_name, depth_limit=predecessor_depth_limit, tree=True)
        successors = self.successors(table_name, depth_limit=successors_depth_limit, tree=True)

        table_links_graph = graphviz.Digraph(format='png', graph_attr={'rankdir': 'LR'})
        table_links_graph.node(table_name, fillcolor='lightcoral', style='filled')

        for successor, predecessor_group in predecessors.items():
            for predecessor in predecessor_group:
                table_links_graph.edge(predecessor, successor)
        for predecessor, successor_group in successors.items():
            for successor in successor_group:
                table_links_graph.edge(predecessor, successor)

        return table_links_graph

    def _read_graph_from_filesystem(self):
        graph = nx.DiGraph()
        graph.add_node(self.REFRESH_DATETIME_NODE, time=str(dt.datetime.now()))

        all_tables = Filesystem.all_tables()
        if self._is_jupyter():
            all_tables = tqdm(all_tables)
        for table in all_tables:
            graph.add_node(table)
            for dependency in TableInFilesystem(table).extract_all_dependencies(pass_over_views=False):
                graph.add_edge(dependency, table)
        return graph

    def _read_graph_from_cache(self):
        if self.cache_path.exists():
            cache = json.loads(self.cache_path.read_text())
            graph = json_graph.node_link_graph(cache)
            return graph
        else:
            print('DependencyGraph Сache does not exist\n' 'DependencyGraph.refresh() is highly recomended')
            return None

    def _write_graph_to_cache(self, graph):
        cache = json_graph.node_link_data(graph)
        self.cache_path.write_text(json.dumps(cache))

    def _alert_graph_expiration(self, graph):
        refresh_datetime = graph.nodes[self.REFRESH_DATETIME_NODE]['time']
        refresh_datetime = dt.datetime.fromisoformat(refresh_datetime)
        refresh_age = dt.datetime.now() - refresh_datetime
        if refresh_age > self.refresh_delay:
            print(f'DependencyGraph is expired ({refresh_age})\n' 'DependencyGraph.refresh() is advised')

    def _is_jupyter(self):
        try:
            get_ipython()
            return True
        except NameError:
            return False

    def _recursive_successors_or_predesessors(self, table_name, depth_left, method='successors'):
        method_callable = getattr(self.graph, method)
        successors = {table_name: list(method_callable(table_name))}
        if depth_left > 1:
            for successor in successors[table_name]:
                successor_successors = self._recursive_successors_or_predesessors(
                    successor, depth_left=depth_left - 1, method=method
                )
                if len(successor_successors[successor]) > 0:
                    successors = {**successors, **successor_successors}
        return successors
