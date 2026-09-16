# agents/mappers/eval_mapper.py 
import os
import torch
import numpy as np

from agents.mappers.mapper_direct import DirectMapper
from agents.mappers.mapper_residual import ResidualMapper


def load_mapper(model_path=None, residual=False):
    """Charge un mapper entraîné"""
    if model_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if residual:
            model_path = os.path.join(
                project_root, "data", "DIRECT_PPO", "residual", "checkpoints", "mapper_residual.pt"
            )
        else:
            model_path = os.path.join(
                project_root, "data", "DIRECT_PPO", "complete", "checkpoints", "mapper_direct.pt"
            )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Mapper checkpoint not found at {model_path}")
    
    # Créer le modèle avec les bonnes dimensions
    if residual:
        print("Loading RESIDUAL mapper")
        model = ResidualMapper()
    else:
        print("Loading DIRECT mapper")
        model = DirectMapper()
    
    # Charger les poids
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Adapter les clés si nécessaire (pour la compatibilité)
    # Certains modèles sont sauvegardés avec "mlp.0.weight" au lieu de "mlp.net.0.weight"
    new_state_dict = {}
    for key, value in state_dict.items():
        # Si les clés sont "mlp.0.weight" mais le modèle attend "mlp.net.0.weight"
        if key.startswith("mlp.") and not key.startswith("mlp.net."):
            new_key = key.replace("mlp.", "mlp.net.", 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Charger avec strict=False pour ignorer les clés manquantes
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def predict(s2, model=None, residual=False):
    """Prédit s3 à partir de s2"""
    if model is None:
        model = load_mapper(residual=residual)
    
    # s2 doit avoir 6 dimensions pour 2DoF
    if len(s2.shape) == 1:
        s2 = s2.reshape(1, -1)
    
    s2 = torch.tensor(s2, dtype=torch.float32)
    
    with torch.no_grad():
        s3 = model(s2)
    
    return s3.squeeze(0).numpy()


def evaluate_mappers():
    """Évalue et compare les deux mappers"""
    print("=" * 60)
    print("EVALUATING DIRECT MAPPER")
    print("=" * 60)
    
    # Charger le mapper direct
    model_direct = load_mapper(residual=False)
    
    # Charger le mapper résiduel
    print("\n" + "=" * 60)
    print("EVALUATING RESIDUAL MAPPER")
    print("=" * 60)
    model_residual = load_mapper(residual=True)
    
    # Tester avec quelques états
    test_states = [
        np.array([0.1, -0.2, 0.05, -0.03, 0.5, -0.3], dtype=np.float32),  # État aléatoire
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),       # État neutre
        np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0], dtype=np.float32),       # État extrême
        np.array([-1.0, -1.0, -0.5, -0.5, -1.0, -1.0], dtype=np.float32), # État extrême négatif
    ]
    
    print("\n" + "-" * 60)
    print("TEST PREDICTIONS")
    print("-" * 60)
    
    for i, s2 in enumerate(test_states):
        print(f"\nTest {i+1}: s2 = {s2}")
        
        # Prédiction directe
        s3_direct = predict(s2, model_direct, residual=False)
        print(f"  Direct mapper   -> s3 = {s3_direct}")
        
        # Prédiction résiduelle
        s3_residual = predict(s2, model_residual, residual=True)
        print(f"  Residual mapper -> s3 = {s3_residual}")
        
        # Différence entre les deux
        diff = np.abs(s3_direct - s3_residual)
        print(f"  Différence      = {diff}")
        print(f"  MSE             = {np.mean(diff**2):.6f}")


if __name__ == "__main__":
    # Test rapide avec le mapper direct par défaut
    print("=" * 60)
    print("QUICK TEST - DIRECT MAPPER")
    print("=" * 60)
    
    model = load_mapper(residual=False)
    dummy = np.array([0.1, -0.2, 0.05, -0.03, 0.5, -0.3], dtype=np.float32)
    out = predict(dummy, model, residual=False)
    print(f"Input s2:  {dummy}")
    print(f"Pred s3:  {out}")
    
    # Évaluer les deux mappers
    evaluate_mappers()
