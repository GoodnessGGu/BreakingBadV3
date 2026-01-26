import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from nn_utils import BinaryOptionsNN

# =========================
# CONFIG
# =========================
CSV_PATH = "training_data.csv"
BATCH_SIZE = 128
EPOCHS = 500
LR = 0.001
EARLY_STOPPING_PATIENCE = 500
LR_PATIENCE = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# TRAIN FUNCTION
# =========================
def train_nn_for_expiry(expiry, X_scaled, df):
    print(f"\nTraining {expiry} minute model...")

    outcome_col = f"outcome_{expiry}m"
    if outcome_col not in df.columns:
        print(f"Skipping {expiry}m: {outcome_col} not found in CSV.")
        return

    y = torch.tensor(df[outcome_col].values, dtype=torch.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = BinaryOptionsNN(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    
    # Scheduler: Reduce learning rate when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=LR_PATIENCE, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    save_path = f"model_{expiry}m.pth"

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            X_val_dev = X_val.to(device)
            y_val_dev = y_val.to(device)
            val_pred_prob = model(X_val_dev).squeeze()
            val_loss = loss_fn(val_pred_prob, y_val_dev)
            total_val_loss = val_loss.item()
            
            # Metrics
            val_pred = (val_pred_prob > 0.5).float()
            val_acc = (val_pred == y_val_dev).float().mean()

        # Update Scheduler
        scheduler.step(total_val_loss)

        # Early Stopping & Best Model Saving
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            status_symbol = "* (Best)"
        else:
            epochs_no_improve += 1
            status_symbol = ""

        if (epoch + 1) % 10 == 0 or status_symbol:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {total_val_loss:.4f} | Val Acc: {val_acc.item():.4f} {status_symbol}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training Complete for {expiry}m. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run collect_data.py first.")
        exit(1)

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    # drop non-feature columns
    drop_cols = ["signal", "pair", "asset", "market_type", "from", "to", "time", "outcome", "outcome_1m", "outcome_5m"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    print(f"Features used: {list(X.columns)}")

    # scale once (important!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")
    print("Saved scaler.pkl")

    X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Train both
    train_nn_for_expiry(1, X_scaled_tensor, df)
    train_nn_for_expiry(5, X_scaled_tensor, df)
