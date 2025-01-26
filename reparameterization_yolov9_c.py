import torch
from models.yolo import Model
from pathlib import Path


def reparameterize_yolov9_c(ckpt_path, nc: int):
    device = torch.device("cpu")
    cfg = "./models/detect/gelan-c.yaml"
    model = Model(cfg, ch=3, nc=80, anchors=3)

    model = model.to(device)
    _ = model.eval()

    # Load the checkpoint from file
    ckpt_data = torch.load(ckpt_path, map_location='cpu')  # Keep the original path intact
    model.names = ckpt_data['model'].names
    model.nc = ckpt_data['model'].nc

    idx = 0
    for k, v in model.state_dict().items():
        if f"model.{idx}." in k:
            if idx < 22:
                kr = k.replace(f"model.{idx}.", f"model.{idx + 1}.")
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt_data['model'].state_dict()[kr]
            elif f"model.{idx}.cv2." in k:
                kr = k.replace(f"model.{idx}.cv2.", f"model.{idx + 16}.cv4.")
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt_data['model'].state_dict()[kr]
            elif f"model.{idx}.cv3." in k:
                kr = k.replace(f"model.{idx}.cv3.", f"model.{idx + 16}.cv5.")
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt_data['model'].state_dict()[kr]
            elif f"model.{idx}.dfl." in k:
                kr = k.replace(f"model.{idx}.dfl.", f"model.{idx + 16}.dfl2.")
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt_data['model'].state_dict()[kr]
        else:
            while True:
                idx += 1
                if f"model.{idx}." in k:
                    break
            if idx < 22:
                kr = k.replace(f"model.{idx}.", f"model.{idx + 1}.")
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt_data['model'].state_dict()[kr]
            elif f"model.{idx}.cv2." in k:
                kr = k.replace(f"model.{idx}.cv2.", f"model.{idx + 16}.cv4.")
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt_data['model'].state_dict()[kr]
            elif f"model.{idx}.cv3." in k:
                kr = k.replace(f"model.{idx}.cv3.", f"model.{idx + 16}.cv5.")
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt_data['model'].state_dict()[kr]
            elif f"model.{idx}.dfl." in k:
                kr = k.replace(f"model.{idx}.dfl.", f"model.{idx + 16}.dfl2.")
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt_data['model'].state_dict()[kr]
    _ = model.eval()

    m_ckpt = {
        'model': model.half(),
        'optimizer': None,
        'best_fitness': None,
        'ema': None,
        'updates': None,
        'opt': None,
        'git': None,
        'date': None,
        'epoch': -1,
    }

    # Use the original file path to construct the new output path
    output_path = Path(ckpt_path).parent / "best-converted.pt"
    torch.save(m_ckpt, output_path)
    print(f"Converted model saved to {output_path}")


if __name__ == "__main__":
    ckpt_list = [
        "./runs/train/yolov9-80epochs/weights/best.pt",
        "./runs/train/yolov9-160epochs/weights/best.pt",
    ]
    for ckpt in ckpt_list:
        reparameterize_yolov9_c(ckpt, nc=1)
