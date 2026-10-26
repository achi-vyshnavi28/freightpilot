import os
import tempfile

# Must be set before any app module reads settings.
_tmp = tempfile.mkdtemp(prefix="fp-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["FP_LLM_MODE"] = "offline"
os.environ.pop("FP_API_KEYS", None)

import pytest  # noqa: E402

from app.db import Base, engine, init_db  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    init_db()
    yield
