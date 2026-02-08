import importlib

try:
    torch = importlib.import_module("torch")
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))
        print("cuda_version:", torch.version.cuda)
    else:
        print("cuda_version:", torch.version.cuda)
except Exception as exc:
    print("torch_import_error:", type(exc).__name__, str(exc))