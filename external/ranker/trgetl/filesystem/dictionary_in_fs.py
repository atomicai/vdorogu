import xml.etree.ElementTree as xmltree

from .filesystem import DICT_PATH, Filesystem


class DictionaryInFilesystem(Filesystem):
    def __init__(self, name):
        self.name = name

    def usage_examples(self, column=None):
        metadata = self.extract_metadata()

        key = metadata["key"]
        if isinstance(key, str):
            key = f"toUInt64({key})"
        else:
            key = "(" + ", ".join(key) + ")"

        examples = {attr: f"dictGet('{self.name}', '{attr}', {key})" for attr in metadata["attributes"]}
        if column is None:
            return examples
        else:
            return examples.get(column)

    def extract_metadata(self):
        root = self._get_xml_root()
        metadata = {}

        metadata["table_name"] = self.table_name(root)
        metadata["key"] = self.key(root)
        metadata["attributes"] = self.attributes(root)

        return metadata

    def attributes(self, root=None):
        if root is None:
            root = self._get_xml_root()
        return [attr.findtext("name") for attr in root.find("structure").findall("attribute")]

    def key(self, root=None):
        if root is None:
            root = self._get_xml_root()

        id = root.find("structure").find("id")
        key = root.find("structure").find("key")
        assert id or key, "Dictionary structure has neither id nor key"

        if id:
            return id.findtext("name")
        if key:
            return [attr.findtext("name") for attr in key.findall("attribute")]

    def table_name(self, root=None):
        if root is None:
            root = self._get_xml_root()
        schema = root.find("source").find("clickhouse").findtext("db")
        table = root.find("source").find("clickhouse").findtext("table")
        return f"{schema}.{table}"

    def metadata_path(self):
        return self._find_query_path(
            self.name,
            DICT_PATH,
        )

    def _get_xml_root(self):
        metadata_path = self.metadata_path()
        tree = xmltree.parse(metadata_path)
        root = tree.getroot()
        return root
