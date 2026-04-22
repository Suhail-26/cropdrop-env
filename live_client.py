import requests
from typing import Dict, Any, Tuple

class CropdropLiveClient:
    def __init__(self, base_url: str = "https://suhailma-cropdrop-env.hf.space"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reset(self) -> Dict[str, Any]:
        resp = self.session.post(f"{self.base_url}/reset")
        resp.raise_for_status()
        data = resp.json()
        if 'observation' in data:
            return data['observation']
        return data

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # ✅ Wrap the action inside {"action": ...}
        resp = self.session.post(f"{self.base_url}/step", json={"action": action})
        resp.raise_for_status()
        data = resp.json()
        obs = data.get('observation', {})
        reward = data.get('reward', 0.0)
        done = data.get('done', False)
        info = data.get('info', {})
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.session.close()