import sys
from pathlib import Path

# Add src and .venv site-packages to path
src_path = Path("src").absolute()
sys.path.insert(0, str(src_path))
venv_site_packages = Path(".venv/Lib/site-packages").absolute()
sys.path.insert(0, str(venv_site_packages))

try:
    from power_grid_model_io.converters.tabular_converter import TabularConverter
    print("TabularConverter imported successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
