import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def get_current_username(
    credentials: HTTPBasicCredentials = Depends(security)
):
    correct_user = os.getenv("BASIC_AUTH_USERNAME", "admin")
    correct_pass = os.getenv("BASIC_AUTH_PASSWORD", "secret")
    if not (
        secrets.compare_digest(credentials.username, correct_user)
        and secrets.compare_digest(credentials.password, correct_pass)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
