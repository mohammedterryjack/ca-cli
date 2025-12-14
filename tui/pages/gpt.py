from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, Input
from textual.containers import Horizontal, Vertical, ScrollableContainer
from pathlib import Path
from tools.utils import load_model, generate_text, line_plot
from rich.text import Text
from tools.utils import BOS


# --------------------------------------------------------
# Generate page: all CLI options
# --------------------------------------------------------
class GPTGenerateView(Screen):
    def compose(self):
        yield Header()
        yield Static("CharGPT Generate", classes="subtitle")

        with Horizontal():
            for label, id_, default in [
                ("Input Dir", "input_dir", "path/to/model_dir"),
                ("Checkpoint", "checkpoint", "final.pt"),
                ("Prompt", "prompt", BOS),
                ("Temperature", "temperature", "1.0"),
                ("Top-k", "top_k", ""),
                ("Output Dir", "output_dir", "path/to/save_dir"),
            ]:
                with Vertical():  # one row
                    yield Static(label, classes="opt-label")  # left
                    yield Input(value=str(default), id=id_)   # right
        
        with Horizontal():
            yield Button("Run", id="run")
            yield Button("Back", id="back")

        yield Static("", id="output_box", classes="output")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "run":
            self.run_generate()

    def run_generate(self):
        output_box = self.query_one("#output_box", Static)
        output_box.update("")
        try:
            input_dir = Path(self.query_one("#input_dir", Input).value)
            checkpoint = self.query_one("#checkpoint", Input).value
            prompt = self.query_one("#prompt", Input).value
            temperature = float(self.query_one("#temperature", Input).value)
            top_k_str = self.query_one("#top_k", Input).value
            top_k = int(top_k_str) if top_k_str.strip() != "" else None
            output_dir_str = self.query_one("#output_dir", Input).value
            output_dir = Path(output_dir_str) if output_dir_str else None

            output_box.update("[green]Loading model…[/green]")

            model, vocab = load_model(input_dir, checkpoint)
            generated = generate_text(
                model=model,
                vocab=vocab,
                prompt=prompt,
                length=model.config.sequence_len,
                temperature=temperature,
                top_k=top_k,
            )
            output_box.update(f"[green]=== Generated Text ===[/green]\n{generated}")

        except Exception as e:
            output_box.update(f"[red]Error: {e}[/red]")

