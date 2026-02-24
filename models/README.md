# AI PCB Assistant - Models Directory

This directory contains pre-trained AI models.

## Required Models

1. **LLM Model** (GGUF format)
   - Download: https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF
   - File: `deepseek-coder-6.7b-instruct.Q5_K_M.gguf`
   - Size: ~5GB

2. **Placement Model** (ONNX format)
   - File: `placement_model.onnx`
   - Size: ~50MB
   - Note: Will be trained and distributed separately

## Download Instructions

```bash
# Download DeepSeek Coder GGUF
curl -L -o deepseek-coder-6.7b-instruct.Q5_K_M.gguf \
  "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q5_K_M.gguf"
```

Or use the `download_models.py` script in `ai_backend/`.
