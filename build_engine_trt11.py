import tensorrt as trt
import sys

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
config = builder.create_builder_config()

network = builder.create_network()
parser = trt.OnnxParser(network, logger)

onnx_file = "yolov8m.onnx"
engine_file = "yolov8m.engine"

print(f"📦 Đang đọc file {onnx_file}...")
with open(onnx_file, "rb") as f:
    if not parser.parse(f.read()):
        print("❌ Lỗi đọc file ONNX:")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        sys.exit(1)

# Bật trực tiếp cờ FP16 cho TensorRT 11.x
config.set_flag(trt.BuilderFlag.FP16)

# Cấp phát 2 GB Workspace tạm thời cho TensorRT 11.x
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

print("🚀 Đang biên dịch yolov8m.onnx sang yolov8m.engine FP16 (TensorRT 11)...")
print("⚡ Quá trình này có thể mất từ 1 - 3 phút, vui lòng đợi...")

serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("❌ Biên dịch Engine thất bại!")
else:
    with open(engine_file, "wb") as f:
        f.write(serialized_engine)
    print(f"\n✅ THÀNH CÔNG! File đã được lưu tại: {engine_file}")
