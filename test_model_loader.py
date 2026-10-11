from model_loader import load_model

for name in ["SVM", "Random Forest"]:
    try:
        model = load_model(name)
        print(f"[SUCCESS] {name}: โหลดโมเดลสำเร็จ")
        print(f"  Class: {type(model)}")
        print(f"  Feature set: {getattr(model, 'feature_set', 'ไม่พบข้อมูล')}")
        print(f"  Threshold: {getattr(model, 'threshold', 'ไม่พบข้อมูล')}")
    except Exception as e:
        print(f"[FAILED] {name}: {type(e).__name__}: {e}")
