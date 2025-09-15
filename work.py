import wasmtime

store = wasmtime.Store()
module = wasmtime.Module.from_file(store.engine, 'mcstream.wasm')

for imp in module.imports:
    print(f"Import: module={imp.module}, name={imp.name}, type={imp.type}")



# Build imports dict (example, replace with your actual imports)
imports = {
    "env": {
        "memory": wasmtime.Memory(store, wasmtime.MemoryType(minimum=256)),
        "table": wasmtime.Table(store, wasmtime.TableType(wasmtime.FuncType([], []), 0)),
        "abort": lambda: print("abort called"),
        # Add all other imports here with correct signatures
    },
    # Add other import modules if needed
}

instance = wasmtime.Instance(store, module, [imports["env"]])
