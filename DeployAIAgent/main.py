import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from agent import run_agent

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# Global exception handlers to ensure JSON responses for errors
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    content = {"detail": exc.detail if hasattr(exc, "detail") else str(exc)}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log the exception server-side and return a generic JSON error
    import logging
    logging.exception("Unhandled server error:")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Request model
class AgentRequest(BaseModel):
    """Request model for agent invocation."""
    prompt: str


# Response model
class AgentResponse(BaseModel):
    """Response model for agent invocation."""
    response: str


@app.get("/")
async def home(request: Request):
    """Serve the main HTML interface."""
    # return templates.TemplateResponse("index.html", {"request": request})
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={} # You can add other variables here
    )


@app.post("/agent", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """
    Invoke the AI agent with a prompt.

    The agent can read and write text files based on natural language instructions.
    """
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Run the agent with the user's prompt
        result = run_agent(request.prompt)

        return AgentResponse(response=result)

    except HTTPException:
        raise
    except Exception as e:
        # Let the global exception handler convert this to JSON
        raise


@app.get("/routes")
async def list_routes():
    """Diagnostic endpoint: list registered routes and methods."""
    routes = []
    for route in app.router.routes:
        # Some route types may not have 'methods' or 'path'
        path = getattr(route, 'path', None) or getattr(route, 'pattern', None)
        methods = list(getattr(route, 'methods', []))
        name = getattr(route, 'name', None)
        routes.append({"path": str(path), "methods": methods, "name": name})
    return routes


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
