"""원본의 저장 로그만 시각화하며 학습을 실행하지 않습니다."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "results/recorded_metrics.json").read_text(encoding="utf-8"))
history = data["history"]
epochs = [r["epoch"] for r in history]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
axes[0].plot(epochs, [r["train_loss"] for r in history], color="#2563EB", marker="o", linewidth=2.3)
axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy loss")
axes[1].plot(epochs, [r["test_accuracy_percent"] for r in history], color="#0D9488", marker="o", linewidth=2.3)
axes[1].set(title="Known-command accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 107))
for ax in axes:
    ax.set_xticks(epochs)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Whisper small | 11 Korean commands", fontsize=15, fontweight="bold", y=.99)
fig.text(.5,.015,"Saved logs: Train n=99, Test n=55. No new training run. Unknown detection is not measured here.",ha="center",fontsize=8.5,color="#475569")
fig.tight_layout(rect=(0,.07,1,.94))
target=ROOT/"docs/assets/training_history.png"
target.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(target,dpi=180,bbox_inches="tight")
print(target)
