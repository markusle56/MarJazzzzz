from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import symusic
from miditok import TokSequence
import data_preparation.featuriser as featuriser
from data_preparation.prepare_dataset import PreTokenizedDataset, TRAIN_PATH, VAL_PATH, TEST_PATH

# --- Hyperparameters ---
EMBED_DIM   = 128
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
MLP_DIM     = 256
DROPOUT     = 0.3
BATCH_SIZE  = 32
MAX_SEQ_LEN = 1024
NUM_FILES   = 1000
EPOCHS      = 50
LR          = 1e-3
LR_FACTOR   = 0.5
LR_PATIENCE = 2
ES_PATIENCE = 5
MAX_GEN_LEN = 500
SEED_PATH   = "A2_seed.mid"
OUT_PATH    = "generated.mid"
CSV_PATH    = "filtered_midi.csv"
MODEL_PATH  = "marjazz.pth"

vocab_size = len(featuriser.tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MarJazz(nn.Module):
    # embedding → LSTM → Dropout → LayerNorm
    #           → LSTM → Dropout → LayerNorm
    #           → MLP  → Dropout → LayerNorm
    #           → Linear → logits

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        mlp_dim: int = MLP_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm1 = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.mlp   = nn.Linear(hidden_size, mlp_dim)
        self.act   = nn.GELU()
        self.drop3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(mlp_dim)

        self.fc = nn.Linear(mlp_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        x, _ = self.lstm1(x)
        x = self.norm1(self.drop1(x))

        x, _ = self.lstm2(x)
        x = self.norm2(self.drop2(x))

        x = self.act(self.mlp(x))
        x = self.norm3(self.drop3(x))

        return self.fc(x)


def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=featuriser.tokenizer.pad_token_id)
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            inputs, targets = input_ids[:, :-1], input_ids[:, 1:]
            outputs = model(inputs)
            total_loss += criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1)).item()
    return total_loss / len(dataloader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
    lr_factor: float = LR_FACTOR,
    lr_patience: int = LR_PATIENCE,
    es_patience: int = ES_PATIENCE,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lr_factor, patience=lr_patience
    )
    criterion = nn.CrossEntropyLoss(ignore_index=featuriser.tokenizer.pad_token_id)
    best_val_loss, no_improve = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            inputs, targets = input_ids[:, :-1], input_ids[:, 1:]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        val_loss   = evaluate(model, val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}")
        if val_loss < best_val_loss:
            best_val_loss, no_improve = val_loss, 0
        else:
            no_improve += 1
            if no_improve >= es_patience:
                print(f"Early stopping at epoch {epoch+1} (val loss no improvement for {es_patience} epochs)")
                break


def generate(model: nn.Module, seed_path: str, max_length: int = MAX_GEN_LEN) -> list[int | list[int]]:
    model.eval()
    score = symusic.Score(seed_path)
    generated = cast(TokSequence, featuriser.tokenizer.encode(score)[0]).ids[:]
    with torch.no_grad():
        for _ in range(max_length):
            inputs = torch.tensor(generated).unsqueeze(0).to(device)
            outputs = model(inputs)
            next_token = int(torch.argmax(outputs[0, -1]).item())
            generated.append(next_token)
    return generated


def save_midi(tokens: list[int | list[int]], output_path: str):
    score = featuriser.tokenizer.decode([TokSequence(ids=tokens)])
    score.dump_midi(output_path)


def save_model(model: nn.Module, path: str = MODEL_PATH):
    torch.save({
        "model_state": model.state_dict(),
        "hyperparams": {
            "vocab_size": vocab_size,
            "embed_dim":  EMBED_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "mlp_dim":    MLP_DIM,
            "dropout":    DROPOUT,
        },
    }, path)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_PATH) -> nn.Module:
    checkpoint = torch.load(path, weights_only=True, map_location=device)
    model = MarJazz(**checkpoint["hyperparams"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model


def main():
    print(f"Using device: {device}")
    train_loader = DataLoader(PreTokenizedDataset(TRAIN_PATH), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PreTokenizedDataset(VAL_PATH),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(PreTokenizedDataset(TEST_PATH),  batch_size=BATCH_SIZE)

    model = MarJazz(vocab_size).to(device)
    train(model, train_loader, val_loader)
    save_model(model)

    test_loss = evaluate(model, test_loader)
    print(f"Test loss: {test_loss:.4f}")

    generated_tokens = generate(model, SEED_PATH)
    save_midi(generated_tokens, OUT_PATH)


if __name__ == "__main__":
    main()