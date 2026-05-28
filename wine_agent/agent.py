from __future__ import annotations

from pathlib import Path

import pandas as pd


class WineAnalystAgent:
    def __init__(self, output_dir: str | Path = 'wine_agent/outputs'):
        self.output_dir = Path(output_dir)
        self.metrics = pd.read_csv(self.output_dir / 'wine_training_metrics.csv')
        self.importance = pd.read_csv(self.output_dir / 'wine_feature_importance.csv')

    def answer(self, question: str) -> str:
        query = question.lower().strip()
        top_features = self.importance.head(5)
        best_row = self.metrics.sort_values('val_roc_auc', ascending=False).iloc[0]

        if 'high-quality' in query or 'predict' in query or 'strongly' in query:
            feature_lines = [
                f"- {row.feature}: importance drop {row.importance_drop:.4f}"
                for row in top_features.itertuples(index=False)
            ]
            return (
                f"The best validation ROC-AUC reached {best_row.val_roc_auc:.3f} with validation accuracy "
                f"{best_row.val_accuracy:.3f}. The strongest predictors of high-quality wine in this model were:\n"
                + '\n'.join(feature_lines)
                + "\n\nPractical reading: alcohol and acidity-related features usually matter most; the exact ranking is in the saved importance CSV."
            )

        if 'model' in query or 'performance' in query or 'accuracy' in query:
            return (
                f"The model's best validation ROC-AUC was {best_row.val_roc_auc:.3f} and the associated validation accuracy was {best_row.val_accuracy:.3f}."
            )

        return (
            "I can answer questions about model performance, top predictors of high-quality wine, and which chemical factors were most important. "
            "Try asking: 'What chemical factors most strongly predict high-quality wine?'"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Wine analytical agent')
    parser.add_argument('--question', required=True)
    parser.add_argument('--outputs', default='wine_agent/outputs')
    args = parser.parse_args()

    agent = WineAnalystAgent(args.outputs)
    print(agent.answer(args.question))


if __name__ == '__main__':
    main()
