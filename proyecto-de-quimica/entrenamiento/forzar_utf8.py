import sys
sys.path.insert(0, ".")
# Fuerza codificación UTF-8 en Windows para caracteres Unicode
sys.stdout = open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False)
sys.stderr = open(sys.stderr.fileno(), 'w', encoding='utf-8', closefd=False)

from entrenamiento import reentrenar
import sys as sys_mod
sys_mod.exit(reentrenar.main())