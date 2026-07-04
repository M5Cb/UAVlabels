"""
Test if AutoTracker works as a standalone module
测试AutoTracker是否可以作为独立模块工作
"""
import os
import sys

# 仅添加当前目录到path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

print(f"Current directory: {_current_dir}")
print(f"sys.path[0]: {sys.path[0]}")

try:
    print("\n[TEST] Importing config...")
    from config import AutoTrackerConfig
    print("  [OK] AutoTrackerConfig imported")

    print("\n[TEST] Importing yolo_exporter...")
    from yolo_exporter import YOLOExporter
    print("  [OK] YOLOExporter imported")

    print("\n[TEST] Importing preprocessor...")
    from preprocessor import DeviceAwarePreprocessor
    print("  [OK] DeviceAwarePreprocessor imported")

    print("\n[TEST] Importing model_builder...")
    from model_builder import build_model, load_checkpoint
    print("  [OK] model_builder functions imported")

    print("\n[TEST] Importing tracker_core...")
    from tracker_core import AutoTracker
    print("  [OK] AutoTracker imported")

    print("\n[SUCCESS] All imports successful!")
    print("\nAutoTracker is now a standalone module that can be copied to other projects.")
    print(f"\nYou can copy the entire 'auto_tracker' folder to another project and it will work.")
    print(f"No need for external lib/ folder!")

except Exception as e:
    print(f"\n[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
