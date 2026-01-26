from pathlib import Path
from textual.screen import Screen
from textual.widgets import Header, Static, Button, Input, ProgressBar
from textual.containers import Horizontal, Vertical
from tools.utils import (
    load_model,
    generate_text,
    evolution_to_parquet_string,
    safe_string_to_evolution,
    save_evolutions_as_parquet,
    EOL,
)
import traceback


class FFNNPredictView(Screen):
    """Screen for loading trained weights and generating predictions."""

    def compose(self):
        yield Header()
        yield Static("FFNN / Generative Prediction", classes="subtitle")

        # Input grid
        with Horizontal():
            with Vertical():
                yield Static("Input Parquet", classes="opt-label")
                yield Input(id="input_path", placeholder="path/to/input.parquet")

            with Vertical():
                yield Static("Weights File", classes="opt-label")
                yield Input(id="weights_path", placeholder="path/to/weights.pt")

            with Vertical():
                yield Static("Learn Mode", classes="opt-label")
                # fallback: just an Input box for now
                yield Input(
                    id="learn_mode", value="generative", placeholder="generative"
                )

        # Buttons
        with Horizontal():
            yield Button("Run Prediction", id="run")
            yield Button("Back", id="back")

        # Progress bar and output
        yield ProgressBar(id="predict_progress", total=100, show_percentage=True)
        yield Static("", id="output_box", classes="output")

    # -----------------------------
    # Button Events
    # -----------------------------
    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "run":
            self.start_prediction()

    # -----------------------------
    # RUN PREDICTION
    # -----------------------------
    def start_prediction(self):
        output_box = self.query_one("#output_box", Static)
        output_box.update("[green]Starting prediction…[/green]")

        input_path = Path(self.query_one("#input_path", Input).value)
        weights_path = Path(self.query_one("#weights_path", Input).value)
        learn_mode = self.query_one("#learn_mode", Input).value.strip().lower()

        if learn_mode != "generative":
            output_box.update(
                f"[red]Only 'generative' mode supported for prediction[/red]"
            )
            return

        if not input_path.is_file() or input_path.suffix != ".parquet":
            output_box.update(f"[red]Invalid input file: {input_path}[/red]")
            return
        if not weights_path.is_file():
            output_box.update(f"[red]Invalid weights file: {weights_path}[/red]")
            return

        try:
            # Load model and vocab
            model, vocab = load_model(weights_path, "final.pt")

            # Load input evolutions
            from tools.utils import load_evolutions_from_parquet

            loaded_evolutions = load_evolutions_from_parquet(input_path)

            predicted_evolutions = {}

            # Iterate through each input evolution
            for name, evo in loaded_evolutions.items():
                output_box.update(f"[yellow]Predicting for {name}…[/yellow]")
                X_img_str = evolution_to_parquet_string(evo[:-1])
                prompt = X_img_str.split(EOL)[0] + EOL

                Y_raw = generate_text(
                    model=model,
                    vocab=vocab,
                    prompt=prompt,
                    length=model.config.sequence_len,
                    temperature=1.0,
                    top_k=None,
                )

                try:
                    Y_hat = safe_string_to_evolution(Y_raw)
                    if not any(Y_hat):
                        raise Exception("No prediction generated")
                    Y_hat = Y_hat[1:]
                except Exception as e:
                    output_box.update(f"[red]Prediction failed: {e}[/red]")
                    Y_hat = None

                if Y_hat is not None:
                    predicted_evolutions[name] = Y_hat

            # Save predictions
            save_dir = input_path.parent / f"{input_path.stem}_predictions"
            save_dir.mkdir(exist_ok=True)
            save_evolutions_as_parquet(predicted_evolutions, output_dir=save_dir)
            output_box.update(f"[green]Predictions saved to {save_dir}[/green]")

        except Exception as e:
            tb = traceback.format_exc()
            output_box.update(
                f"[red]Error during prediction:[/red]\n{e}\n\n[grey]{tb}[/grey]"
            )
