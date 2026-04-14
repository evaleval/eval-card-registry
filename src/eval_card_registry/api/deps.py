from fastapi import Depends, HTTPException

from eval_card_registry.config import settings


def check_writable():
    if settings.read_only:
        raise HTTPException(status_code=405, detail="Write operations disabled in read-only mode")


writable = [Depends(check_writable)]
