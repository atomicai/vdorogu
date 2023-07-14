import json

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from tqdm.notebook import tqdm
except ImportError:
    tqdm = iter

from ..filesystem import REPO_PATH, Filesystem, TableInFilesystem


class DependencyGraph:
    def __init__(self):
        self.graph = self._dependency_graph()

    def __call__(self):
        return self.graph

    def refresh(self):
        all_tables = Filesystem.all_live_tables()
        metadata_dict = {}
        for table_name in tqdm(all_tables):
            table = TableInFilesystem(table_name)
            db_class, db = table.get_db()
            dependencies = list(table.extract_live_dependencies())
            metadata_dict[table_name] = {"operator_type": db_class, "dependencies": dependencies}
        json_path = REPO_PATH / "dags" / "trgetl-main-loader-metadata.json"
        json_path.write_text(json.dumps(metadata_dict, indent=4))
        print(len(metadata_dict), "tables found")

        all_views = Filesystem.all_views()
        metadata_dict = {}
        for view_name in tqdm(all_views):
            view = TableInFilesystem(view_name)
            dependencies = list(view.extract_live_dependencies())
            metadata_dict[view_name] = {"dependencies": dependencies}
        json_path = REPO_PATH / "dags" / "views-metadata.json"
        json_path.write_text(json.dumps(metadata_dict, indent=4))
        print(len(metadata_dict), "views found")

        self.graph = self._dependency_graph()

    def all_live_tables(self):
        tables = list(nx.topological_sort(self.graph))
        return tables

    def dependents(self, tname, recursive=False):
        if recursive:
            deps = list(nx.single_source_shortest_path(self.graph, tname).keys())
            deps = sorted(set(deps))
            deps.remove(tname)
        else:
            deps = []
            metadata_path = REPO_PATH / "dags" / "trgetl-main-loader-metadata.json"
            metadata = json.loads(metadata_path.read_text())
            for dep_name, parameters in metadata.items():
                if tname in parameters["dependencies"]:
                    deps.append(dep_name)
            view_metadata_path = REPO_PATH / "dags" / "views-metadata.json"
            view_metadata = json.loads(view_metadata_path.read_text())
            for view, parameters in view_metadata.items():
                if tname in parameters["dependencies"]:
                    deps.append(view)
            deps = sorted(deps)
        return deps

    def _dependency_graph(self):
        graph = nx.DiGraph()
        metadata_path = REPO_PATH / "dags" / "trgetl-main-loader-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        for table_name, parameters in metadata.items():
            graph.add_node(table_name)
            for dep in parameters["dependencies"]:
                graph.add_edge(dep, table_name)
        return graph
