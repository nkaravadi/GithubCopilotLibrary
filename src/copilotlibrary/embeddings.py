"""
Embedding support via GitHub Copilot REST API.

The copilot-language-server itself doesn't expose embedding endpoints over
JSON-RPC, but it stores the same OAuth token it uses for its own API calls
in a local config file (~/.config/github-copilot/apps.json).  We read that
token automatically, exchange it for a short-lived session token via the
Copilot internal token endpoint, and call the embeddings REST API — all
without any manual token setup, exactly like how the language server works
for chat.

Usage::

    from copilotlibrary.embeddings import CopilotEmbeddings

    emb = CopilotEmbeddings()
    result = emb.embed("Hello world")
    print(f"Dimensions: {result.dimensions}")   # 1536

    # Batch — single API call
    results = emb.embed_batch(["Hello", "World"])
"""
from __future__ import annotations

import json
import os
import platform
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


# ── Token discovery ───────────────────────────────────────────────────────────

def _apps_json_path() -> str:
    system = platform.system()
    if system == "Windows":
        base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "github-copilot", "apps.json")
    return os.path.expanduser("~/.config/github-copilot/apps.json")


def _hosts_json_path() -> str:
    system = platform.system()
    if system == "Windows":
        base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "github-copilot", "hosts.json")
    return os.path.expanduser("~/.config/github-copilot/hosts.json")


def find_copilot_token() -> str | None:
    """Return the locally stored Copilot OAuth token, if available.

    Checks in order:

    1. ``COPILOT_TOKEN`` / ``GITHUB_TOKEN`` environment variables
    2. ``~/.config/github-copilot/apps.json``   (written by Copilot CLI / IntelliJ)
    3. ``~/.config/github-copilot/hosts.json``  (written by some older tooling)
    """
    for env_var in ("COPILOT_TOKEN", "GITHUB_TOKEN"):
        val = os.getenv(env_var)
        if val:
            return val

    apps_path = _apps_json_path()
    if os.path.isfile(apps_path):
        try:
            with open(apps_path, encoding="utf-8") as f:
                data = json.load(f)
            copilot_token: str | None = None
            fallback_token: str | None = None
            for _key, entry in data.items():
                tok = entry.get("oauth_token") or entry.get("token")
                if not tok:
                    continue
                app_id = entry.get("githubAppId", "")
                if app_id.startswith("Ov23li"):
                    copilot_token = tok
                else:
                    fallback_token = fallback_token or tok
            return copilot_token or fallback_token
        except (json.JSONDecodeError, OSError):
            pass

    hosts_path = _hosts_json_path()
    if os.path.isfile(hosts_path):
        try:
            with open(hosts_path, encoding="utf-8") as f:
                data = json.load(f)
            entry = data.get("github.com", {})
            tok = entry.get("oauth_token") or entry.get("token")
            if tok:
                return tok
        except (json.JSONDecodeError, OSError):
            pass

    return None


def _exchange_for_session_token(oauth_token: str) -> str:
    """Exchange a Copilot OAuth token for a short-lived session token.

    The session token is required by ``api.githubcopilot.com`` (the same
    exchange the language server performs internally before every API call).
    """
    req = urllib.request.Request(
        "https://api.github.com/copilot_internal/v2/token",
        headers={
            "Authorization": f"token {oauth_token}",
            "Accept": "application/json",
            "User-Agent": "GitHubCopilotChat/0.1",
            "Editor-Version": "JetBrains-IDE/2025.1",
            "Editor-Plugin-Version": "copilotlibrary/0.2.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        token = data.get("token")
        if not token:
            raise RuntimeError(f"No token in exchange response: {data}")
        return token
    except urllib.error.HTTPError as exc:
        details = ""
        try:
            details = exc.read().decode()
        except Exception:
            pass
        raise RuntimeError(
            f"Token exchange failed (HTTP {exc.code}): {details}"
        ) from exc


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """Result from an embedding request."""
    vector: list[float]
    model: str
    input_text: str
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        return len(self.vector)


# ── Client ────────────────────────────────────────────────────────────────────

class CopilotEmbeddings:
    """Access embeddings via the GitHub Copilot REST API.

    Zero-config: automatically reads the Copilot OAuth token written by
    IntelliJ / Copilot CLI to ``~/.config/github-copilot/apps.json``,
    exchanges it for a session token, and calls the embeddings endpoint.
    No manual token setup required — same experience as chat.

    You can still override with the ``COPILOT_TOKEN`` / ``GITHUB_TOKEN``
    environment variables or pass ``token=`` explicitly.

    Example::

        from copilotlibrary.embeddings import CopilotEmbeddings

        emb = CopilotEmbeddings()
        result = emb.embed("def binary_search(arr, target): ...")
        print(f"Vector ({result.dimensions} dims): {result.vector[:3]}...")

        # Batch — all texts in a single API call
        results = emb.embed_batch(["hello", "world", "code"])
    """

    BASE_URL = "https://api.githubcopilot.com"
    AVAILABLE_MODELS: list[str] = [
        "text-embedding-3-small",   # 1536 dims — default; confirmed available
        "text-embedding-ada-002",   # 1536 dims — legacy; confirmed available
        # "text-embedding-3-large" — 3072 dims; requires higher Copilot plan
    ]
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """
        Args:
            token: Override the auto-discovered Copilot OAuth token.
            base_url: Override the API base URL (useful for proxies / enterprise).
            model: Default embedding model.
        """
        self._oauth_token = token or find_copilot_token()
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.model = model

        # Session token cache: (value, expiry_epoch)
        self._session_token: str | None = None
        self._session_expires: float = 0.0

        if not self._oauth_token:
            raise ValueError(
                "No Copilot token found.\n"
                "Options:\n"
                "  1. Sign in to GitHub Copilot in IntelliJ IDEA — zero config.\n"
                "  2. Set the COPILOT_TOKEN environment variable.\n"
                "  3. Pass token= to CopilotEmbeddings()."
            )

    def _get_session_token(self) -> str:
        """Return a valid session token, refreshing if necessary."""
        now = time.time()
        if self._session_token and now < self._session_expires - 60:
            return self._session_token
        self._session_token = _exchange_for_session_token(self._oauth_token)  # type: ignore[arg-type]
        # Session tokens typically last 30 minutes
        self._session_expires = now + 1800
        return self._session_token

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, text: str, model: str | None = None) -> EmbeddingResult:
        """Get the embedding vector for *text*.

        Args:
            text: Text to embed.
            model: Override the instance-level default model.

        Returns:
            :class:`EmbeddingResult` with ``.vector``, ``.dimensions``,
            ``.model``, and ``.usage``.
        """
        return self.embed_batch([text], model=model)[0]

    def embed_batch(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[EmbeddingResult]:
        """Get embedding vectors for multiple texts in a single API call.

        Args:
            texts: List of texts to embed.
            model: Override the instance-level default model.

        Returns:
            List of :class:`EmbeddingResult` objects, one per input text,
            in the same order as *texts*.
        """
        if not texts:
            return []

        model = model or self.model
        session = self._get_session_token()

        payload = json.dumps({"input": texts, "model": model}).encode()
        headers = {
            "Authorization": f"Bearer {session}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Editor-Version": "JetBrains-IDE/2025.1",
            "Editor-Plugin-Version": "copilotlibrary/0.2.0",
        }

        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode()
            except Exception:
                pass
            raise RuntimeError(
                f"Embedding API returned HTTP {exc.code}: {exc.reason}\n{details}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Network error calling embedding API: {exc.reason}"
            ) from exc

        usage: dict[str, int] = body.get("usage", {})
        actual_model: str = body.get("model", model)

        return [
            EmbeddingResult(
                vector=item["embedding"],
                model=actual_model,
                input_text=texts[i] if i < len(texts) else "",
                usage=usage,
            )
            for i, item in enumerate(body.get("data", []))
        ]

    def list_models(self) -> list[str]:
        """Return the list of supported embedding model IDs."""
        return list(self.AVAILABLE_MODELS)


# ── Convenience functions ─────────────────────────────────────────────────────

def get_embedding(
    text: str,
    model: str = CopilotEmbeddings.DEFAULT_MODEL,
    token: str | None = None,
) -> list[float]:
    """Get the embedding vector for *text*.

    Token is auto-discovered from ``~/.config/github-copilot/apps.json``
    — no setup needed if you're signed in to Copilot in IntelliJ.
    """
    return CopilotEmbeddings(token=token, model=model).embed(text).vector


def get_embeddings_batch(
    texts: list[str],
    model: str = CopilotEmbeddings.DEFAULT_MODEL,
    token: str | None = None,
) -> list[list[float]]:
    """Get embedding vectors for multiple texts (single API call).

    Token is auto-discovered from ``~/.config/github-copilot/apps.json``.
    """
    return [
        r.vector
        for r in CopilotEmbeddings(token=token, model=model).embed_batch(texts)
    ]

