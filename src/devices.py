import openvino as ov


def main():
    # 1. Initialize the OpenVINO engine
    core = ov.Core()

    # 2. Ask the engine to detect all accessible Intel AI hardware
    devices = core.available_devices

    if not devices:
        print("No OpenVINO-compatible devices found.")
        return

    print(f"Found {len(devices)} device(s): {', '.join(devices)}\n")
    print("-" * 50)

    # 3. Loop through each detected device (CPU, GPU.0, NPU, etc.)
    for device in devices:
        print(f"Device: **{device}**")

        # 4. Ask the device what properties it supports querying
        try:
            supported_properties = core.get_property(device, "SUPPORTED_PROPERTIES")
        except Exception as e:
            print(f"  [Could not query properties: {e}]\n")
            continue

        # 5. Loop through the supported properties and print their values
        for property_key in supported_properties:
            # We skip the master lists themselves to avoid infinite loops/clutter
            if property_key not in (
                "SUPPORTED_PROPERTIES",
                "SUPPORTED_METRICS",
                "SUPPORTED_CONFIG_KEYS",
            ):
                try:
                    property_val = core.get_property(device, property_key)
                    print(f"  - {property_key}: {property_val}")
                except TypeError:
                    # Some complex internal properties can't be neatly printed as text
                    print(f"  - {property_key}: [UNSUPPORTED TYPE]")

        print("-" * 50)


if __name__ == "__main__":
    main()
