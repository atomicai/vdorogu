import getpass
from pathlib import Path

CH_EVENTSPASS_PASSWORD = None
CH_EVENTSPASS_USER = None
CONFLUENCE_ACCESSTOKEN = None
CONFLUENCE_CONSUMERKEY = None
CONFLUENCE_PATTOKEN = None
CONFLUENCE_PRIVATEKEY = None
EMAIL_ADDRESS = None
MYSQL_TRGDB_DEV_PASSWORD = None
MYSQL_TRGDB_DEV_USER = None
MYTEAMBOT_TOKEN = None
MYTRACKER_API_PASSWORD = None
MYTRACKER_API_USER_ID = None
PG_OPEX_PASSWORD = None
PG_OPEX_USER = None
RELAP_REDASH_API_TOKEN = None
TABLEAU_TOKEN = None


def apply_tokens(**kwargs):
    for key, value in kwargs.items():
        globals()[key.upper()] = value


def apply_default_tokens():
    if getpass.getuser() == 'jenkins-trgan':
        _apply_jenkins_tokens()

    if getpass.getuser() == 'airflow-trgetl':
        _apply_airflow_tokens()

    else:
        default_keys_path = Path(__file__).parent / '__tokens__'
        if default_keys_path.exists():
            for key_file in default_keys_path.glob('*.token'):
                value = key_file.read_text()
                key = key_file.stem.upper()
                globals()[key] = value


def _apply_jenkins_tokens():
    if getpass.getuser() == 'jenkins-trgan':
        myteambot_token_paths = Path('/usr/local/etc').glob('*.token')
        myteambot_token_paths = list(myteambot_token_paths)
        global MYTEAMBOT_TOKEN
        MYTEAMBOT_TOKEN = myteambot_token_paths[0].read_text()


def _apply_airflow_tokens():
    if getpass.getuser() == 'airflow-trgetl':
        try:
            from airflow.models import Variable

            variables = {
                'CH_EVENTSPASS_PASSWORD': 'trgetl_ch_events_password_secret',
                'CH_EVENTSPASS_USER': 'trgetl_ch_events_user_secret',
                'CONFLUENCE_ACCESSTOKEN': 'trgetl_confluence_accesstoken_secret',
                'CONFLUENCE_CONSUMERKEY': 'trgetl_confluence_consumerkey_secret',
                'CONFLUENCE_PATTOKEN': 'trgetl_confluence_pattoken_secret',
                'CONFLUENCE_PRIVATEKEY': 'trgetl_confluence_privatekey_secret',
                'EMAIL_ADDRESS': 'trgetl_email_send_from',
                'MYSQL_TRGDB_DEV_PASSWORD': 'mysql_trgdb_dev_password',
                'MYSQL_TRGDB_DEV_USER': 'mysql_trgdb_dev_user',
                'MYTEAMBOT_TOKEN': 'trgetl_myteam_bot_secret',
                'MYTRACKER_API_PASSWORD': 'trgetl_mytracker_api_password_secret',
                'MYTRACKER_API_USER_ID': 'trgetl_mytracker_api_user_id_secret',
                'PG_OPEX_PASSWORD': 'trgetl_pg_opex_password_secret',
                'PG_OPEX_USER': 'trgetl_pg_opex_user_secret',
                'RELAP_REDASH_API_TOKEN': 'trgetl_relap_redash_api_token_secret',
            }
            for token_name, token_variable in variables.items():
                globals()[token_name] = Variable.get(token_variable, None)
        except ImportError:
            pass


apply_default_tokens()
