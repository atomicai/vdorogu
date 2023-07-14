import pytest

from ...database.hdfs import Hdfs

ACCESS_LOG_PATH = "/dwh/ods/ods_rb_data.db/access_log/source=*"
ANTIFRAUD_PATH = "/dwh/ods/ods_target_data.db/antifraud/dt=2021-09-20/_SUCCESS"
DUMMY_PATH = "/dwh/ods/ods_target_data.db/antifraud/dt=2021-09-20/_DUMMY"


class HdfsFiles:
    def __init__(self, name, path, ls_list):
        self.name = name
        self.path = path
        self.ls_list = ls_list


HDFS_FILES = {
    HdfsFiles(
        "access_log",
        "/dwh/ods/ods_rb_data.db/access_log/source=*/dt=2021-09-20/_SUCCESS",
        [
            "/dwh/ods/ods_rb_data.db/access_log/source=ADP/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=ADQ/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=API/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=HBID/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=MOBILE/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=MVAST/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=OKFEED/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=POSTBACK/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=R_MAIL_RU/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=VAST/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=VKFEED/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=VP/dt=2021-09-20/_SUCCESS",
            "/dwh/ods/ods_rb_data.db/access_log/source=YOULAFEED/dt=2021-09-20/_SUCCESS",
        ],
    ),
    HdfsFiles(
        "antifraud_success",
        "/dwh/ods/ods_target_data.db/antifraud/dt=2021-09-20/_SUCCESS",
        ["/dwh/ods/ods_target_data.db/antifraud/dt=2021-09-20/_SUCCESS"],
    ),
    HdfsFiles("antifraud_dummy", "/dwh/ods/ods_target_data.db/antifraud/dt=2021-09-20/_DUMMY", []),
}


def id_hdfs_file(objects):
    return [o.name for o in objects]


@pytest.fixture(params=HDFS_FILES, ids=id_hdfs_file(HDFS_FILES))
def hdfs_file(request):
    return request.param


def test_ls(hdfs_file):
    result = Hdfs().ls(hdfs_file.path)
    etalon_answer = hdfs_file.ls_list
    assert result == etalon_answer, f"Should be {etalon_answer}, but got {result}"
