"""
Entry point for the Gaussian Splatting API server
This is a wrapper to maintain backward compatibility with the old execution method
"""
if __name__ == "__main__":
    import uvicorn
    from app.config import settings

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
