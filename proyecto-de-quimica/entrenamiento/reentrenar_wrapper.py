import os
import sys
sys.path.insert(0, ".")
os.environ["PYTHONUTF8"] = "1"
from entrenamiento import reentrenar
sys.exit(reentrenar.main())