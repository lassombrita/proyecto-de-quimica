#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera para_revision.zip con los archivos de verificación del proyecto.
"""

import zipfile
from pathlib import Path

def crear_zip_para_revision():
    """Crea el archivo ZIP con los resultados de verificación."""
    
    carpeta_para_revision = Path("para_revision")
    ruta_zip = Path("para_revision.zip")
    directorio_base = Path.cwd()
    
    print("=" * 60)
    print("📦 GENERANDO para_revision.zip")
    print("=" * 60)
    
    # Verificar contenido actual
    if carpeta_para_revision.exists():
        archivos = list(carpeta_para_revision.rglob('*'))
        print(f"\n✅ Archivos en para_revision/ ({len(archivos)}):")
        for a in sorted(archivos):
            if a.is_file():
                print(f"   📄 {a.relative_to(directorio_base)}")
    else:
        print(f"\n❌ Carpeta 'para_revision/' no existe.")
        return False
    
    # Crear ZIP
    with zipfile.ZipFile(ruta_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        print("\n📦 Agregando archivos al ZIP...")
        for filepath in sorted(carpeta_para_revision.rglob('*')):
            if filepath.is_file():
                arcname = filepath.relative_to(directorio_base)
                print(f"   ➕ {arcname}")
                zf.write(filepath, arcname=arcname)
    
    # Verificar resultado
    size_bytes = ruta_zip.stat().st_size
    size_mb = size_bytes / (1024**2)
    
    print("\n" + "=" * 60)
    print("✅ para_revision.zip generado con éxito!")
    print("=" * 60)
    print(f"\n📍 Ruta completa: {ruta_zip.absolute()}")
    print(f"📊 Tamaño: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    
    # Listar contenido del ZIP
    with zipfile.ZipFile(ruta_zip, 'r') as zf:
        print(f"\n📋 Contenido del ZIP:")
        for item in sorted(zf.namelist()):
            info = zf.getinfo(item)
            comp_size = info.compress_size
            ratio = comp_size / info.file_size * 100 if info.file_size > 0 else 0
            print(f"   • {item:45} ({comp_size:,} bytes, {ratio:.0f}% compression)")
    
    print("\n🎉 ¡Todo listo para la revisión!")
    return True

if __name__ == "__main__":
    crear_zip_para_revision()