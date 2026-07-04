"""
Verify that AutoTracker module is properly set up
验证AutoTracker模块是否正确安装
"""
import os
import sys
from pathlib import Path

# Add project root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def check_lib_modules():
    """Check if all required lib modules exist"""
    required_modules = [
        'lib/models/ortrack',
        'lib/models/layers',
        'lib/test/tracker',
        'lib/test/utils',
        'lib/train/data',
        'lib/utils',
    ]

    print('[CHECK] Checking required lib modules...')
    all_exist = True
    for module in required_modules:
        path = os.path.join(_project_root, module.replace('/', os.sep))
        exists = os.path.exists(path)
        status = '[OK]' if exists else '[FAIL]'
        print(f'  {status} {module}')
        if not exists:
            all_exist = False

    return all_exist


def check_auto_tracker_files():
    """Check if AutoTracker module files exist"""
    required_files = [
        'auto_tracker/__init__.py',
        'auto_tracker/config.py',
        'auto_tracker/model_builder.py',
        'auto_tracker/tracker_core.py',
        'auto_tracker/yolo_exporter.py',
        'auto_tracker/run_auto_track.py',
    ]

    print('\n[CHECK] Checking AutoTracker files...')
    all_exist = True
    for file in required_files:
        path = os.path.join(_project_root, file.replace('/', os.sep))
        exists = os.path.exists(path)
        status = '[OK]' if exists else '[FAIL]'
        print(f'  {status} {file}')
        if not exists:
            all_exist = False

    return all_exist


def check_checkpoint():
    """Check if checkpoint file exists"""
    print('\n[CHECK] Checking checkpoint file...')
    checkpoint_path = os.path.join(_project_root, 'ORTrack_ep0300.pth.tar')
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f'  [OK] ORTrack_ep0300.pth.tar ({size_mb:.1f} MB)')
        return True
    else:
        print(f'  [FAIL] ORTrack_ep0300.pth.tar not found')
        return False


def check_python_packages():
    """Check if required Python packages are installed"""
    print('\n[CHECK] Checking Python packages...')
    required_packages = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('easydict', 'EasyDict'),
    ]

    all_installed = True
    for package_name, display_name in required_packages:
        try:
            __import__(package_name)
            print(f'  [OK] {display_name}')
        except ImportError:
            print(f'  [FAIL] {display_name} (not installed)')
            all_installed = False

    return all_installed


def check_imports():
    """Test if AutoTracker modules can be imported"""
    print('\n[CHECK] Testing module imports...')

    try:
        from auto_tracker.config import AutoTrackerConfig
        print('  [OK] AutoTrackerConfig')
    except Exception as e:
        print(f'  [FAIL] AutoTrackerConfig: {e}')
        return False

    try:
        from auto_tracker.yolo_exporter import YOLOExporter
        print('  [OK] YOLOExporter')
    except Exception as e:
        print(f'  [FAIL] YOLOExporter: {e}')
        return False

    try:
        from auto_tracker.model_builder import build_model, load_checkpoint
        print('  [OK] model_builder (build_model, load_checkpoint)')
    except Exception as e:
        print(f'  [FAIL] model_builder: {e}')
        return False

    try:
        from auto_tracker.tracker_core import AutoTracker
        print('  [OK] AutoTracker')
    except Exception as e:
        print(f'  [FAIL] AutoTracker: {e}')
        return False

    try:
        from auto_tracker import AutoTracker, AutoTrackerConfig
        print('  [OK] Module __init__ exports')
    except Exception as e:
        print(f'  [FAIL] Module __init__: {e}')
        return False

    return True


def main():
    """Run all checks"""
    print('='*60)
    print('AutoTracker Setup Verification')
    print('='*60)

    results = {
        'lib_modules': check_lib_modules(),
        'auto_tracker_files': check_auto_tracker_files(),
        'checkpoint': check_checkpoint(),
        'python_packages': check_python_packages(),
        'imports': check_imports(),
    }

    print('\n' + '='*60)
    print('Summary:')
    print('='*60)

    all_ok = all(results.values())

    for check, result in results.items():
        status = '[OK]' if result else '[FAIL]'
        print(f'{status} {check}')

    print('='*60)

    if all_ok:
        print('\n[SUCCESS] All checks passed! AutoTracker is ready to use.')
        print('\nQuick start:')
        print('  1. Edit auto_tracker/run_auto_track.py')
        print('  2. Set IMAGE_DIR, OUTPUT_DIR, INIT_BBOX, CHECKPOINT_PATH')
        print('  3. Run: python auto_tracker/run_auto_track.py')
    else:
        print('\n[WARNING] Some checks failed. Please address the issues above.')

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
