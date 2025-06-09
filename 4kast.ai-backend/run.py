# run.py
import uvicorn
 
if __name__ == "__main__":
    # the string "app.main:app" points Uvicorn to your FastAPI instance
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=5000,
        reload=True,       # auto-reload on code changes (dev only)
    )
 
# run.py
#from app.main import app  
 
#if __name__ == "__main__":
#    import uvicorn
#    import os
#    port = int(os.environ.get("PORT", 8080))
 #   uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
 
 
   
   
#    uvicorn.run(
#        "app.main:app",
#       host="127.0.0.1",
#        port=5000,
#        reload=True,       # auto-reload on code changes (dev only)
#    )