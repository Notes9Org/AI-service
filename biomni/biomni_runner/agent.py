"""BioMniAgent wrapper around biomni.agent.A1, configured for AWS Bedrock and local data."""
import matplotlib
matplotlib.use("Agg")

import time
from pathlib import Path
from typing import Optional, Tuple, List

from biomni_runner.config import (
    ensure_bedrock_env,
    get_data_path,
    get_llm_model,
    get_timeout_seconds,
)


class BioMniAgent:
    """Wrapper for BioMni A1 agent with Bedrock and local data lake configuration."""

    def __init__(
        self,
        path: Optional[str] = None,
        llm: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        load_datalake: bool = True,
    ):
        """Initialize BioMni agent.

        Args:
            path: Parent of biomni_data/ (BioMni uses path/biomni_data/data_lake). Defaults to BIOMNI_DATA_PATH or biomni/data/biomni.
            llm: Bedrock model ID. Defaults to BIOMNI_LLM_MODEL or backend BEDROCK_CHAT_MODEL_ID.
            timeout_seconds: Timeout for agent tasks. Defaults to BIOMNI_TIMEOUT_SECONDS or 600.
            load_datalake: If True, load datalake from local path (default). If False, skip loading (faster init, some tools won't work).
        """
        ensure_bedrock_env()
        raw_path = path or get_data_path()
        self._path = str(Path(raw_path).resolve())
        self._llm = llm or get_llm_model()
        self._timeout = timeout_seconds or get_timeout_seconds()
        self._load_datalake = load_datalake
        self._agent = None

    def _get_agent(self):
        """Lazy-initialize the A1 agent."""
        if self._agent is None:
            # Set default_config BEFORE creating A1 so internal tools
            # (database queries, data lake lookups) use the correct settings
            from biomni.config import default_config
            default_config.path = self._path
            default_config.source = "Bedrock"
            default_config.llm = self._llm
            default_config.timeout_seconds = self._timeout

            from biomni.agent import A1
            agent_kwargs = {
                "path": self._path,
                "llm": self._llm,
                "source": "Bedrock",
                "commercial_use_allowed": True,
            }
            if not self._load_datalake:
                agent_kwargs["expected_data_lake_files"] = []
            self._agent = A1(**agent_kwargs)
        return self._agent

    def go(self, query: str, max_retries: int = 3) -> Tuple[List[str], str]:
        """Execute a biomedical task and return the response.

        Retries automatically on Bedrock throttling errors with exponential backoff.

        Args:
            query: Natural language task (e.g. "Predict ADMET properties for aspirin").
            max_retries: Max retries on throttling errors (default 3).

        Returns:
            Tuple of (log_steps, final_answer).
        """
        agent = self._get_agent()
        for attempt in range(max_retries + 1):
            try:
                return agent.go(query)
            except Exception as e:
                if "ThrottlingException" in str(e) and attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    print(f"Bedrock throttled, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                raise
