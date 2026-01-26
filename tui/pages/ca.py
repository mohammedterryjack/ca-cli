from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, Input
from textual.containers import Horizontal, Vertical
from tools.utils import CellularAutomata, save_evolution, save_evolutions_as_parquet
from rich.text import Text
from pathlib import Path
from tui.utils import render_halfblock_rich


# --------------------------------------------------------
# Parse IC like CLI version
# --------------------------------------------------------
def parse_ic(ic_value: str | None):
    if ic_value is None or ic_value.strip() == "":
        return None

    ic_value = ic_value.strip().lower()

    if ic_value.startswith("x") and ic_value[1:].isdigit():
        return ("random", int(ic_value[1:]))

    if ":" in ic_value:
        a, b = ic_value.split(":")
        return list(range(int(a), int(b) + 1))

    return [int(ic_value)]


# --------------------------------------------------------
#                   MAIN SCREEN
# --------------------------------------------------------
class CAView(Screen):
    def compose(self):
        yield Header()
        yield Static("Cellular Automata", classes="subtitle")

        # -----------------------------
        # Input row (labels + inputs)
        # -----------------------------
        with Horizontal():
            for label, id_, placeholder in [
                ("IC", "ic", "None"),
                ("Rule", "rule", "0"),
                ("Neighbourhood", "neighbourhood", "1"),
                ("Width", "width", "30"),
                ("Timesteps", "timesteps", "20"),
                ("States", "states", "2"),
                ("Output Dir", "output", ""),
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
        # Error + Output Panels
        # -----------------------------
        yield Static("", id="ca_output", classes="output")

    # --------------------------------------------------------
    # Button-handling
    # --------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "run":
            self.run_ca()

    # --------------------------------------------------------
    # Run CA with error handling + safe parsing
    # --------------------------------------------------------
    def run_ca(self):

        output_box = self.query_one("#ca_output", Static)
        # Clear both
        output_box.update("")

        try:
            # -----------------------------
            # Collect inputs
            # -----------------------------
            ic_input = self.query_one("#ic", Input).value or None
            rule = int(self.query_one("#rule", Input).value or 0)
            neighbourhood = int(self.query_one("#neighbourhood", Input).value or 1)
            width = int(self.query_one("#width", Input).value or 30)
            timesteps = int(self.query_one("#timesteps", Input).value or 20)
            states = int(self.query_one("#states", Input).value or 2)
            output_input = self.query_one("#output", Input).value

            output_dir = Path(output_input) if output_input else None
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)

            # -----------------------------
            # Parse IC list
            # -----------------------------
            parsed_ic = parse_ic(ic_input)

            if parsed_ic is None:
                ic_list = [None]
            elif isinstance(parsed_ic, tuple):
                _, count = parsed_ic
                ic_list = [None] * count
            else:
                ic_list = parsed_ic

            # -----------------------------
            # Run & render
            # -----------------------------
            output_text = Text()
            evolutions = {}

            for ic in ic_list:
                ca = CellularAutomata(
                    cell_states=states,
                    neighbourhood_radius=neighbourhood,
                    lattice_width=width,
                    time_steps=timesteps,
                    initial_state=ic,
                    transition_rule_number=rule,
                )

                output_text.append(render_halfblock_rich(ca.evolution))

                # Save if output dir selected
                if output_dir:
                    actual_ic = ca.info.lattice_evolution[0]
                    actual_rule = ca.info.local_transition_rule
                    filename = f"{actual_rule}-{actual_ic}"
                    save_evolution(ca.evolution, output_dir / f"{filename}.png")
                    evolutions[filename] = ca.evolution

            if output_dir:
                save_evolutions_as_parquet(evolutions, output_dir)

            output_box.update(output_text)

        except Exception as e:
            # Display clean error message
            output_box.update(f"[red]Error: {e}[/red]")
