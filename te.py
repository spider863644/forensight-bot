from forensics import verify_signature

file_path = "reports/run_20251007_233351/manifest.json"
signature_path = file_path + ".sig"
public_key_path = "keys/Spider Anongreyhat_public.pem"

is_valid = verify_signature(file_path, signature_path, public_key_path)

if is_valid:
    print("✅ Signature is valid — file is authentic and untampered.")
else:
    print("❌ Signature verification failed — file may be modified or key mismatch.")
