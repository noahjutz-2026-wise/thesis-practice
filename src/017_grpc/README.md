# gRPC + Unity

## Quickstart

```bash
uv run --with grpcio-tools python -m grpc_tools.protoc -I=protos --python_out=src --pyi_out=src --grpc_python_out=src protos/grpc_unity/generated/messages.proto
```
