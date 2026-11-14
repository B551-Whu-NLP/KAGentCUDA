from data_module.data_loader import DataLoader
import sys
from pathlib import Path
# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python data_loader.py <paired_data_path> [--refresh-dwarf]")
        sys.exit(1)

    base_path = Path(sys.argv[1])
    refresh   = '--refresh-dwarf' in sys.argv

    loader = DataLoader(base_path, obfuscate_code=True)

    if refresh:
        # 只更新矮人向量，不重新解析代码
        if not loader.load_cache():
            loader.load()
        loader.refresh_dwarf_and_save()
    else:
        if not loader.load_cache():
            loader.load()
            loader.save_cache()

    print(f"\nC codes:    {len(loader.c_codes)}")
    print(f"CUDA codes: {len(loader.cuda_codes)}")

    if loader.c_codes:
        c = loader.c_codes[0]
        print(f"\nFirst C loop:")
        print(f"  ID:           {c['id']}")
        print(f"  Project:      {c['project']}")
        print(f"  File:         {c['filename']}")
        print(f"  dwarf_vector: {c['dwarf_vector']}")
        print(f"  Code snippet: {c['code'][:80]}...")

    if loader.cuda_codes:
        k = loader.cuda_codes[0]
        print(f"\nFirst CUDA kernel:")
        print(f"  ID:           {k['id']}")
        print(f"  Project:      {k['project']}")
        print(f"  kernel_name:  {k['kernel_name']}")
        print(f"  dwarf_vector: {k['dwarf_vector']}")
        print(f"  Code snippet: {k['code'][:80]}...")