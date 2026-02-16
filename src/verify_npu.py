import openvino as ov

try:
    core = ov.Core()
    devices = core.available_devices

    print("--- System Check ---")
    print(f"OpenVINO Version: {ov.get_version()}")
    print(f"Available Devices: {devices}")

    if "NPU" in devices:
        # Get the full name of the NPU for your Ultra 285H
        full_name = core.get_property("NPU", "FULL_DEVICE_NAME")
        print(f"✅ SUCCESS: NPU Detected as '{full_name}'")

        # Test a tiny dummy compilation to ensure the compiler is working
        print("Testing NPU compiler...")
        # (Assuming you have a tiny model.xml or using a dummy)
        # core.compile_model(model, "NPU")
    else:
        print("❌ FAILURE: NPU not found in available devices.")
        print("Hint: Check 'dmesg | grep intel_vpu' for firmware errors.")

except Exception as e:
    print(f"An error occurred: {e}")
