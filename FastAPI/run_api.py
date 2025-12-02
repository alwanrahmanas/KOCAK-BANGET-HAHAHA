from pyngrok import ngrok
import uvicorn

# buka tunnel ke port 8000
public_url = ngrok.connect(8000)
print("🔗 NGROK PUBLIC URL:", public_url)

# jalankan fastapi
uvicorn.run("main:app", host="0.0.0.0", port=8000)
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "FastAPI.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # 🔥 MATIKAN RELOAD
        workers=1
    )
