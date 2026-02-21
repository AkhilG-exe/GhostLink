# 👻 GhostLink Scalper

**GhostLink** is a lightweight, high-frequency scalping bot built for the Alpaca Markets API. It uses "Ghost" logic to monitor price action silently and trigger trades based on confirmed pivot point patterns, keeping you informed via Slack webhooks.



---

## 🛠 Key Components

* **The Scalper Engine**: Uses a configurable lookback window to find local price extremes (Pivots).
* **Pattern Recognition**: Detects classic technical setups:
    * **Double Tops/Bottoms**: Price rejection at previous levels.
    * **Head & Shoulders (and Inverse)**: Multi-pivot trend exhaustion.
* **The Link (Webhooks)**: 
    * `Heartbeat`: Monitoring channel for bot status, startup/shutdown, and runtime errors.
    * `Alpha`: Dedicated channel for trade signals, entries, and exits.

---

## 🚦 Getting Started

### 1. Installation
Clone the repository and install the `requests` library:
```bash
git clone [https://github.com/yourusername/GhostLink.git](https://github.com/yourusername/GhostLink.git)
cd GhostLink
pip install requests
```
Enjoy!
