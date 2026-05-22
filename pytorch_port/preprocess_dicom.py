import os
import glob
import numpy as np
import cv2

try:
    import pydicom
except ImportError:
    print("ERROR: pydicom is not installed. Please run: pip install pydicom")
    exit(1)

def main():
    base_dir = "../PROSTATEx-v1-doiJNLP/prostatex"
    out_dir = "../data/prostatex_processed"
    adc_out = os.path.join(out_dir, "adc")
    t2_out = os.path.join(out_dir, "t2")
    
    os.makedirs(adc_out, exist_ok=True)
    os.makedirs(t2_out, exist_ok=True)
    
    patients = [d for d in os.listdir(base_dir) if d.startswith("ProstateX")]
    print(f"Found {len(patients)} patients in dataset.")
    
    paired_names = []
    
    for patient in sorted(patients):
        patient_path = os.path.join(base_dir, patient)
        
        # Traverse down to find series
        # Structure: Patient -> Study -> Series -> Instances (.dcm)
        t2_series = None
        adc_series = None
        
        for root, dirs, files in os.walk(patient_path):
            dcms = [f for f in files if f.endswith(".dcm")]
            if len(dcms) > 0:
                # Read the first dicom to get the series description
                dcm_path = os.path.join(root, dcms[0])
                try:
                    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    desc = str(ds.SeriesDescription).lower()
                    
                    if "t2" in desc and "tra" in desc: # axial T2
                        t2_series = root
                    elif "adc" in desc:
                        adc_series = root
                except Exception as e:
                    pass
        
        if t2_series and adc_series:
            # We found a pair for this patient
            print(f"Processing {patient}...")
            
            # Process T2
            t2_files = sorted(glob.glob(os.path.join(t2_series, "*.dcm")))
            if len(t2_files) == 0: continue
            # Get middle slice
            t2_mid = t2_files[len(t2_files)//2]
            ds_t2 = pydicom.dcmread(t2_mid)
            img_t2 = ds_t2.pixel_array.astype(float)
            
            # Process ADC
            adc_files = sorted(glob.glob(os.path.join(adc_series, "*.dcm")))
            if len(adc_files) == 0: continue
            adc_mid = adc_files[len(adc_files)//2]
            ds_adc = pydicom.dcmread(adc_mid)
            img_adc = ds_adc.pixel_array.astype(float)
            
            # Normalize and crop
            def process_img(img):
                # Normalize to 0-255
                img = img - np.min(img)
                if np.max(img) > 0:
                    img = img / np.max(img) * 255.0
                img = img.astype(np.uint8)
                
                # Center crop to 64x64
                h, w = img.shape
                cy, cx = h//2, w//2
                
                # Ensure it's at least 64x64, else pad
                if h < 64 or w < 64:
                    img = cv2.resize(img, (64, 64))
                else:
                    img = img[cy-32:cy+32, cx-32:cx+32]
                return img
            
            t2_final = process_img(img_t2)
            adc_final = process_img(img_adc)
            
            # Save
            filename = f"{patient}.png"
            cv2.imwrite(os.path.join(t2_out, filename), t2_final)
            cv2.imwrite(os.path.join(adc_out, filename), adc_final)
            paired_names.append(filename)

    print(f"\nSuccessfully extracted {len(paired_names)} paired images.")
    
    # Save lists
    with open(os.path.join(out_dir, "paired_names.txt"), "w") as f:
        for name in paired_names:
            f.write(name + "\n")
            
    with open(os.path.join(out_dir, "t2_names.txt"), "w") as f:
        for name in paired_names:
            f.write(name + "\n")
            
    with open(os.path.join(out_dir, "adc_names.txt"), "w") as f:
        for name in paired_names:
            f.write(name + "\n")

if __name__ == "__main__":
    main()
