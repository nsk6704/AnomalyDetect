import torch
from models import LSTMAutoencoder

def test_lstm():
    seq_len = 64
    n_features = 1
    batch_size = 32
    
    print("Initializing LSTM...")
    model = LSTMAutoencoder(seq_len, n_features)
    model.to('cpu')
    
    x = torch.randn(batch_size, seq_len, n_features)
    print(f"Input shape: {x.shape}")
    
    try:
        y = model(x)
        print(f"Output shape: {y.shape}")
        print("Forward pass successful.")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lstm()
