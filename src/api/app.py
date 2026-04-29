from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch, io
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.models.train import HierDRNet
from src.models.cascade_model import Stage1Net, Stage2Net

app = FastAPI(title="HierDR-Net API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DR_GRADES = {0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe DR", 4: "Proliferative DR"}
TRANSFORM = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_models():
    try:
        m = HierDRNet(pretrained=False).to(DEVICE)
        m.load_state_dict(torch.load("models/hierdr_net.pth", map_location=DEVICE))
        m.eval()
        s1 = Stage1Net(pretrained=False).to(DEVICE)
        s1.load_state_dict(torch.load("models/cascade/stage1_net.pth", map_location=DEVICE))
        s1.eval()
        s2 = Stage2Net(pretrained=False).to(DEVICE)
        s2.load_state_dict(torch.load("models/cascade/stage2_net.pth", map_location=DEVICE))
        s2.eval()
        return m, s1, s2
    except Exception as e:
        return None, None, None

hierdr_model, stage1_model, stage2_model = load_models()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "hierdrnet_loaded": hierdr_model is not None,
        "cascade_loaded": stage1_model is not None,
        "device": str(DEVICE),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if hierdr_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    contents = await file.read()
    img   = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        s1, s2, _ = hierdr_model(tensor)
        s1_pred   = F.softmax(s1, dim=1).argmax().item()
        if s1_pred == 0:
            grade = 0
        else:
            grade = F.softmax(s2, dim=1).argmax().item() + 1
        confidence = {DR_GRADES[i]: round(F.softmax(s2, dim=1)[0][i].item(), 4)
                      for i in range(4)} if s1_pred > 0 else {"No DR": 1.0}
    return {
        "grade":       grade,
        "diagnosis":   DR_GRADES[grade],
        "confidence":  confidence,
    }
