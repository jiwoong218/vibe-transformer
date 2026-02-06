import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformer import Transformer

# Hyperparameters
SRC_VOCAB_SIZE = 5000
TRG_VOCAB_SIZE = 5000
SRC_PAD_IDX = 0
TRG_PAD_IDX = 0
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
MAX_LEN = 100
DROPOUT = 0.1
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_transformer(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for src, trg in progress_bar:
        src, trg = src.to(device), trg.to(device)
        
        # 디코더의 입력은 마지막 단어를 제외한 것: <sos> 단어1 단어2
        # 실제 정답(Label)은 첫 단어를 제외한 것: 단어1 단어2 <eos>
        trg_input = trg[:, :-1]
        trg_real = trg[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward Pass
        # output: (batch_size, trg_len, vocab_size)
        output = model(src, trg_input)
        
        # CrossEntropyLoss를 위해 차원 변경
        # output: (batch_size * trg_len, vocab_size)
        # trg_real: (batch_size * trg_len)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_real = trg_real.contiguous().view(-1)
        
        # Loss 계산 및 역전파
        loss = criterion(output, trg_real)
        loss.backward()
        
        # Gradient Clipping: 기울기 폭주 방지 
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return epoch_loss / len(train_loader)

def main():
    
    # 모델 초기화
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        trg_vocab_size=TRG_VOCAB_SIZE,
        src_pad_idx=SRC_PAD_IDX,
        trg_pad_idx=TRG_PAD_IDX,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        device=DEVICE
    ).to(DEVICE)
    
    # 1. 손실 함수: 패딩은 무시하도록 설정
    # label_smoothing을 주면 성능이 더 잘 나옵니다.
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX, label_smoothing=0.1)

    # 2. 옵티마이저: Adam이 가장 국룰입니다.
    # 베타 값들을 논문과 똑같이 설정하는 디테일!
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # Dummy DataLoader for demonstration
    # (Batch, Seq_Len)
    dummy_src = torch.randint(1, SRC_VOCAB_SIZE, (10, MAX_LEN))
    dummy_trg = torch.randint(1, TRG_VOCAB_SIZE, (10, MAX_LEN))
    
    # 간단한 데이터셋/로더 생성
    from torch.utils.data import TensorDataset, DataLoader
    train_data = TensorDataset(dummy_src, dummy_trg)
    train_loader = DataLoader(train_data, batch_size=2)
    loss = train_transformer(model, train_loader, optimizer, criterion, DEVICE)
    print(f"Training Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
