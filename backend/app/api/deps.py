from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import ValidationError, decode_token
from app.services.container import Container

bearer = HTTPBearer(auto_error=False)


@dataclass
class User:
    name: str
    role: str


def get_container(request: Request) -> Container:
    return request.app.state.container


def current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                 c: Container = Depends(get_container)) -> User:
    if not c.s.auth_enabled:
        return User("dev", "admin")
    if creds is None:
        raise HTTPException(401, "authentication required", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = decode_token(creds.credentials, c.s.jwt_secret)
    except ValidationError:
        raise HTTPException(401, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    return User(claims["sub"], claims.get("role", "viewer"))


def require_admin(u: User = Depends(current_user)) -> User:
    if u.role != "admin":
        raise HTTPException(403, "admin role required")
    return u
