$ErrorActionPreference = "Stop"

# Building the Rust native extension (maturin) requires LLVM/libclang
if (-not $env:LIBCLANG_PATH -and (Test-Path "C:\Program Files\LLVM\bin")) {
    $env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
}

uv sync
uv run uvicorn server:app --port 8081 --host 127.0.0.1
