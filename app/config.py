"""Shared, explicit settings for deterministic backend episodes."""

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEFAULT_START_TIME = datetime(2026, 10, 5, 9, tzinfo=VIETNAM_TZ)
DEFAULT_SEED = 42


class BackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    quote_ttl_seconds: int = Field(default=300, strict=True, gt=0)
    hold_ttl_seconds: int = Field(default=600, strict=True, gt=0)
