from diffuserve.engine.llm_engine import LLMEngine
from diffuserve.engine.dp_engine import DPEngine
from diffuserve.config import Config

class LLM:
    def __new__(cls, model, **kwargs):
        cfg = Config(model, **{k: v for k, v in kwargs.items() if k in Config.__dataclass_fields__.keys()})
        if cfg.data_parallel_size > 1:
            return DPEngine(model, **kwargs)
        return LLMEngine(model, **kwargs)
