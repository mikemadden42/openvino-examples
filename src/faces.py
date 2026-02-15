import os
import time
import urllib.request

import numpy as np
import openvino as ov


def main():
    # 1. Initialize the OpenVINO Engine
    core = ov.Core()

    # 2. Dynamically Find the GPU
    # Look through the available devices for anything with "GPU" in the name
    target_device = None
    full_hardware_name = "Unknown Hardware"

    for device in core.available_devices:
        if "GPU" in device:
            target_device = device  # Usually "GPU" or "GPU.0"
            # Ask the device what its actual commercial product name is
            full_hardware_name = core.get_property(device, "FULL_DEVICE_NAME")
            break

    # Fallback just in case this is run on a machine without a GPU
    if not target_device:
        print("No GPU detected! Gracefully falling back to CPU...")
        target_device = "CPU"
        full_hardware_name = core.get_property("CPU", "FULL_DEVICE_NAME")
    else:
        print(f"Found GPU: {full_hardware_name} (Internal ID: {target_device})")

    # 3. URLs for Intel's pre-trained FP16 Face Detection model
    base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0004/FP16/"
    xml_url = base_url + "face-detection-retail-0004.xml"
    bin_url = base_url + "face-detection-retail-0004.bin"

    # Download the model files directly from Intel if they aren't local
    for filename, url in [("face_model.xml", xml_url), ("face_model.bin", bin_url)]:
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)

    # 4. Read the downloaded model
    print("\nLoading Face Detection model into OpenVINO...")
    model = core.read_model(model="face_model.xml")

    # 5. Compile EXPLICITLY for the dynamically found device
    print(f"Compiling model for the {target_device}...")
    compiled_model = core.compile_model(model=model, device_name=target_device)

    # 6. Create synthetic image data to test the hardware
    input_data = np.random.randint(
        0, 255, size=(1, 3, 300, 300), dtype=np.uint8
    ).astype(np.float32)

    # 7. Warm up the Hardware
    print("Warming up the hardware...")
    compiled_model(input_data)

    # 8. Run the FPS Benchmark
    iterations = 100
    print(f"Running inference stress test ({iterations} frames)...")

    start_time = time.time()
    for _ in range(iterations):
        results = compiled_model(input_data)
    end_time = time.time()

    # 9. Calculate and print the results
    fps = iterations / (end_time - start_time)
    print("\n" + "=" * 50)
    print("                   RESULTS")
    print("+" + "=" * 49)
    print(f" Hardware:  {full_hardware_name}")
    print(" Precision: FP16 (Half-Precision)")
    print(f" Speed:     **{fps:.2f} FPS**")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
