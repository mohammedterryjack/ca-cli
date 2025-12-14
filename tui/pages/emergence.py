from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, Input
from textual.containers import Horizontal, Vertical
from pathlib import Path
from tools.utils import (
    filter_spacetime_by_delay_embedding_density,
    load_evolutions_from_parquet,
    save_evolution,
    save_evolutions_as_parquet,
    render_halfblock,
)
from rich.text import Text
from tui.utils import render_halfblock_rich


class EmergenceView(Screen):
    def compose(self):
        yield Header()
        yield Static("Emergence", classes="subtitle")

        # -----------------------------
        # Input row
        # -----------------------------
        with Horizontal():
            for label, id_, placeholder in [
                ("Input Path", "input_path", "path/to/my_file.parquet"),
                ("Output Dir", "output_path", "path/to/save_dir"),
                ("Binarisation Threshold", "threshold", "0.5"),
                ("Time Delay", "time_delay", "1"),
                ("Embedding Dim", "embedding_dimension", "3"),
            ]:
                with Vertical():
                    yield Static(label, classes="opt-label")
                    yield Input(placeholder=placeholder, id=id_)

        # -----------------------------
        # Buttons
        # -----------------------------
        with Horizontal():
            yield Button("Run", id="run")
            yield Button("Back", id="back")

        # -----------------------------
        # Output / Error Panel
        # -----------------------------
        yield Static("", id="emergence_output", classes="output")

    # --------------------------------------------------------
    # Button-handling
    # --------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "run":
            self.run_emergence()

    # --------------------------------------------------------
    # Run Emergence with error handling
    # --------------------------------------------------------
    def run_emergence(self):
        output_box = self.query_one("#emergence_output", Static)
        output_box.update("")

        try:
            # -----------------------------
            # Collect inputs
            # -----------------------------
            input_path = self.query_one("#input_path", Input).value
            output_path = self.query_one("#output_path", Input).value
            threshold = self.query_one("#threshold", Input).value or "0.5" 
            time_delay = self.query_one("#time_delay", Input).value or "1"
            embedding_dimension = self.query_one("#embedding_dimension", Input).value or "3"

            # Parse values
            input_path = Path(input_path)
            output_path = Path(output_path) if output_path else None
            threshold = float(threshold) if threshold else None
            time_delay = int(time_delay)
            embedding_dimension = int(embedding_dimension)

            if not input_path.is_file() or input_path.suffix != ".parquet":
                raise ValueError(f"Input must be a Parquet file, got {input_path}")

            loaded_evolutions = load_evolutions_from_parquet(input_path)

            output_text = Text()

            if output_path is not None:
                output_path.mkdir(parents=True, exist_ok=True)

            emergence_evolutions = {}


            for name, data in loaded_evolutions.items():
                rareness = filter_spacetime_by_delay_embedding_density(
                    spacetime=data,
                    time_delay=time_delay,
                    embedding_dimension=embedding_dimension,
                )

                if threshold is not None:
                    binary = (rareness >= threshold).astype(int)
                    output_text.append(render_halfblock_rich(binary)) 
                    rareness = binary
                
                if output_path is not None:
                    out_file = output_path / f"{name}.png"
                    save_evolution(values=rareness, path=out_file)
                    output_text.append(f"Saved output image → {out_file}\n\n")
                
                emergence_evolutions[name] = rareness
            
            if output_path is not None:
                save_evolutions_as_parquet(evolutions=emergence_evolutions, output_dir=output_path)
                output_text.append(f"Saved all evolutions to {output_path}.parquet\n")

            output_box.update(output_text)

        except Exception as e:
            output_box.update(f"[red]Error: {e}[/red]")

