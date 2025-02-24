# Copyright (c) Mehmet Bektas <mbektasgh@outlook.com>
#
# GitHub auth and inline completion sections are derivative of https://github.com/B00TK1D/copilot-api

from dataclasses import dataclass
from enum import Enum
import os, json, time, requests, threading
from typing import Any
from pathlib import Path
import uuid
import secrets
import sseclient
import datetime as dt
import logging
from notebook_intelligence.api import CancelToken, ChatResponse, CompletionContext

from ._version import __version__ as NBI_VERSION

log = logging.getLogger(__name__)

EDITOR_VERSION = f"NotebookIntelligence/{NBI_VERSION}"
EDITOR_PLUGIN_VERSION = f"NotebookIntelligence/{NBI_VERSION}"
USER_AGENT = f"NotebookIntelligence/{NBI_VERSION}"
CLIENT_ID = "Iv1.b507a08c87ecfe98"
MACHINE_ID = secrets.token_hex(33)[0:65]

API_ENDPOINT = "https://api.githubcopilot.com"
PROXY_ENDPOINT = "https://copilot-proxy.githubusercontent.com"
TOKEN_REFRESH_INTERVAL = 1500
ACCESS_TOKEN_THREAD_SLEEP_INTERVAL = 5
TOKEN_THREAD_SLEEP_INTERVAL = 3
TOKEN_FETCH_INTERVAL = 15
NL = '\n'
KEYRING_SERVICE_NAME = "NotebookIntelligence"
GITHUB_ACCESS_TOKEN_KEYRING_NAME = "github-copilot-access-token"

LoginStatus = Enum('LoginStatus', ['NOT_LOGGED_IN', 'ACTIVATING_DEVICE', 'LOGGING_IN', 'LOGGED_IN'])

github_auth = {
    "verification_uri": None,
    "user_code": None,
    "device_code": None,
    "access_token": None,
    "status" : LoginStatus.NOT_LOGGED_IN,
    "token": None,
    "token_expires_at": dt.datetime.now()
}

stop_requested = False
get_access_code_thread = None
get_token_thread = None
last_token_fetch_time = dt.datetime.now() + dt.timedelta(seconds=-TOKEN_FETCH_INTERVAL)
remember_github_access_token = False
github_access_token_provided = None

def get_login_status():
    global github_auth

    response = {
        "status": github_auth["status"].name
    }
    if github_auth["status"] is LoginStatus.ACTIVATING_DEVICE:
        response.update({
            "verification_uri": github_auth["verification_uri"],
            "user_code": github_auth["user_code"]
        })

    return response

def login_with_existing_credentials(access_token_config=None):
    global github_access_token_provided, remember_github_access_token

    if github_auth["status"] is not LoginStatus.NOT_LOGGED_IN:
        return

    if access_token_config == "remember" or access_token_config is None:
        try:
            import keyring
            github_access_token_provided = keyring.get_password(KEYRING_SERVICE_NAME, GITHUB_ACCESS_TOKEN_KEYRING_NAME)
        except Exception as e:
            if access_token_config == "remember":
                log.error(f"Failed to get GitHub access token: {e}")
        remember_github_access_token = access_token_config == "remember"
    elif access_token_config == "forget":
        try:
            import keyring
            keyring.delete_password(KEYRING_SERVICE_NAME, GITHUB_ACCESS_TOKEN_KEYRING_NAME)
        except Exception as e:
            log.error(f"Failed to forget GitHub access token: {e}")
    elif access_token_config is not None:
        github_access_token_provided = access_token_config

    if github_access_token_provided is not None:
        login()

def store_github_access_token(access_token):
    if remember_github_access_token:
        try:
            import keyring
            keyring.set_password(KEYRING_SERVICE_NAME, GITHUB_ACCESS_TOKEN_KEYRING_NAME, access_token)
        except Exception as e:
            log.error(f"Failed to store GitHub access token: {e}")

def login():
    login_info = get_device_verification_info()
    if login_info is not None:
        wait_for_tokens()
    return login_info

def logout():
    global github_auth
    github_auth.update({
        "verification_uri": None,
        "user_code": None,
        "device_code": None,
        "access_token": None,
        "status" : LoginStatus.NOT_LOGGED_IN,
        "token": None
    })

    return {
        "status": github_auth["status"].name
    }

def handle_stop_request():
    global stop_requested
    stop_requested = True

def get_device_verification_info():
    global github_auth
    data = {
        "client_id": CLIENT_ID,
        "scope": "read:user"
    }
    try:
        resp = requests.post('https://github.com/login/device/code',
            headers={
                'accept': 'application/json',
                'editor-version': EDITOR_VERSION,
                'editor-plugin-version': EDITOR_PLUGIN_VERSION,
                'content-type': 'application/json',
                'user-agent': USER_AGENT,
                'accept-encoding': 'gzip,deflate,br'
            },
            data=json.dumps(data)
        )

        resp_json = resp.json()
        github_auth["verification_uri"] = resp_json.get('verification_uri')
        github_auth["user_code"] = resp_json.get('user_code')
        github_auth["device_code"] = resp_json.get('device_code')

        github_auth["status"] = LoginStatus.ACTIVATING_DEVICE
    except Exception as e:
        log.error(f"Failed to get device verification info: {e}")
        return None

    # user needs to visit the verification_uri and enter the user_code
    return {
        "verification_uri": github_auth["verification_uri"],
        "user_code": github_auth["user_code"]
    }

def wait_for_user_access_token_thread_func():
    global github_auth, get_access_code_thread

    if github_access_token_provided is not None:
        log.info("Using existing GitHub access token")
        github_auth["access_token"] = github_access_token_provided
        get_access_code_thread = None
        return

    while True:
        # terminate thread if logged out or stop requested
        if stop_requested or github_auth["access_token"] is not None or github_auth["device_code"] is None or github_auth["status"] == LoginStatus.NOT_LOGGED_IN:
            get_access_code_thread = None
            break
        data = {
            "client_id": CLIENT_ID,
            "device_code": github_auth["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
        }
        try:
            resp = requests.post('https://github.com/login/oauth/access_token',
                headers={
                'accept': 'application/json',
                'editor-version': EDITOR_VERSION,
                'editor-plugin-version': EDITOR_PLUGIN_VERSION,
                'content-type': 'application/json',
                'user-agent': USER_AGENT,
                'accept-encoding': 'gzip,deflate,br'
                },
                data=json.dumps(data)
            )

            resp_json = resp.json()
            access_token = resp_json.get('access_token')

            if access_token:
                github_auth["access_token"] = access_token
                get_token()
                get_access_code_thread = None
                store_github_access_token(access_token)
                break
        except Exception as e:
            log.error(f"Failed to get access token from GitHub Copilot: {e}")

        time.sleep(ACCESS_TOKEN_THREAD_SLEEP_INTERVAL)

def get_token():
    global github_auth, github_access_token_provided, API_ENDPOINT, PROXY_ENDPOINT, TOKEN_REFRESH_INTERVAL
    access_token = github_auth["access_token"]

    if access_token is None:
        return

    github_auth["status"] = LoginStatus.LOGGING_IN

    try:
        resp = requests.get('https://api.github.com/copilot_internal/v2/token', headers={
            'authorization': f'token {access_token}',
            'editor-version': EDITOR_VERSION,
            'editor-plugin-version': EDITOR_PLUGIN_VERSION,
            'user-agent': USER_AGENT
        })

        resp_json = resp.json()

        if resp.status_code == 401:
            github_access_token_provided = None
            logout()
            wait_for_tokens()
            return
        
        if resp.status_code != 200:
            log.error(f"Failed to get token from GitHub Copilot: {resp_json}")
            return

        token = resp_json.get('token')
        github_auth["token"] = token
        expires_at = resp_json.get('expires_at')
        if expires_at is not None:
            github_auth["token_expires_at"] = dt.datetime.fromtimestamp(expires_at)
        else:
            github_auth["token_expires_at"] = dt.datetime.now() + dt.timedelta(seconds=TOKEN_REFRESH_INTERVAL)
        github_auth["verification_uri"] = None
        github_auth["user_code"] = None
        github_auth["status"] = LoginStatus.LOGGED_IN

        endpoints = resp_json.get('endpoints', {})
        API_ENDPOINT = endpoints.get('api', API_ENDPOINT)
        PROXY_ENDPOINT = endpoints.get('proxy', PROXY_ENDPOINT)
        TOKEN_REFRESH_INTERVAL = resp_json.get('refresh_in', TOKEN_REFRESH_INTERVAL)
    except Exception as e:
        log.error(f"Failed to get token from GitHub Copilot: {e}")

def get_token_thread_func():
    global github_auth, get_token_thread, last_token_fetch_time
    while True:
        # terminate thread if logged out or stop requested
        if stop_requested or github_auth["status"] == LoginStatus.NOT_LOGGED_IN:
            get_token_thread = None
            return
        token = github_auth["token"]
        # update token if 10 seconds or less left to expiration
        if github_auth["access_token"] is not None and (token is None or (dt.datetime.now() - github_auth["token_expires_at"]).total_seconds() > -10):
            if (dt.datetime.now() - last_token_fetch_time).total_seconds() > TOKEN_FETCH_INTERVAL:
                log.info("Refreshing GitHub token")
                get_token()
                last_token_fetch_time = dt.datetime.now()

        time.sleep(TOKEN_THREAD_SLEEP_INTERVAL)

def wait_for_tokens():
    global get_access_code_thread, get_token_thread
    if get_access_code_thread is None:
        get_access_code_thread = threading.Thread(target=wait_for_user_access_token_thread_func)
        get_access_code_thread.start()

    if get_token_thread is None:
        get_token_thread = threading.Thread(target=get_token_thread_func)
        get_token_thread.start()

def _generate_copilot_headers():
    global github_auth
    token = github_auth['token']

    return {
        'authorization': f'Bearer {token}',
        'editor-version': EDITOR_VERSION,
        'editor-plugin-version': EDITOR_PLUGIN_VERSION,
        'user-agent': USER_AGENT,
        'content-type': 'application/json',
        'openai-intent': 'conversation-panel',
        'openai-organization': 'github-copilot',
        'copilot-integration-id': 'vscode-chat',
        'x-request-id': str(uuid.uuid4()),
        'vscode-sessionid': str(uuid.uuid4()),
        'vscode-machineid': MACHINE_ID,
    }

def inline_completions(prefix, suffix, language, filename, context: CompletionContext, cancel_token: CancelToken) -> str:
    global github_auth
    token = github_auth['token']

    prompt = f"# Path: {filename}"

    if cancel_token.is_cancel_requested:
        return ''

    if context is not None:
        for item in context.items:
            context_file = f"Compare this snippet from {item.filePath if item.filePath is not None else 'undefined'}:{NL}{item.content}{NL}"
            prompt += "\n# " + "\n# ".join(context_file.split('\n'))

    prompt += f"{NL}{prefix}"

    try:
        if cancel_token.is_cancel_requested:
            return ''
        resp = requests.post(f"{PROXY_ENDPOINT}/v1/engines/copilot-codex/completions",
            headers={'authorization': f'Bearer {token}'},
                json={
                'prompt': prompt,
                'suffix': suffix,
                'min_tokens': 500,
                'max_tokens': 2000,
                'temperature': 0,
                'top_p': 1,
                'n': 1,
                'stop': ['<END>', '```'],
                'nwo': 'NotebookIntelligence',
                'stream': True,
                'extra': {
                    'language': language,
                    'next_indent': 0,
                    'trim_by_indentation': True
                }
            }
        )
    except Exception as e:
        log.error(f"Failed to get inline completions: {e}")
        return ''

    if cancel_token.is_cancel_requested:
        return ''

    result = ''

    decoded_response = resp.content.decode()

    resp_text = decoded_response.split('\n')
    for line in resp_text:
        if line.startswith('data: {'):
            json_completion = json.loads(line[6:])
            completion = json_completion.get('choices')[0].get('text')
            if completion:
                result += completion
            # else:
            #     result += '\n'
    
    return result

def completions(messages, tools = None, response: ChatResponse = None, cancel_token: CancelToken = None, options: dict = {}) -> Any:
    stream = response is not None

    try:
        data = {
            'messages': messages,
            'tools': tools,
            'max_tokens': 1000,
            'temperature': 0,
            'top_p': 1,
            'n': 1,
            'stop': ['<END>'],
            'nwo': 'NotebookIntelligence',
            'stream': stream
        }

        if 'tool_choice' in options:
            data['tool_choice'] = options['tool_choice']

        if cancel_token is not None and cancel_token.is_cancel_requested:
            response.finish()

        request = requests.post(
            f"{API_ENDPOINT}/chat/completions",
            headers = _generate_copilot_headers(),
            json = data,
            stream = stream
        )

        if request.status_code != 200:
            msg = f"Failed to get completions from GitHub Copilot: {request.status_code}: {request.text}"
            log.error(msg)
            raise Exception(msg)

        if stream:
            client = sseclient.SSEClient(request)
            for event in client.events():
                if cancel_token is not None and cancel_token.is_cancel_requested:
                    response.finish()
                if event.data == '[DONE]':
                    response.finish()
                else:
                    response.stream(json.loads(event.data))
            return
        else:
            return request.json()
    except requests.exceptions.ConnectionError:
        raise Exception("Connection error")
    except Exception as e:
        log.error(f"Failed to get completions from GitHub Copilot: {e}")
        raise e
