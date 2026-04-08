try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required for the web interface.") from e

try:
    from models import CropdropAction, CropdropObservation
    from server.cropdrop_env_environment import CropdropEnvironment
except ModuleNotFoundError:
    from ..models import CropdropAction, CropdropObservation
    from .cropdrop_env_environment import CropdropEnvironment

def make_env():
    return CropdropEnvironment()

app = create_app(
    make_env,
    CropdropAction,
    CropdropObservation,
    env_name="cropdrop_env",
    max_concurrent_envs=1,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "CropDrop Environment is running"}

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
