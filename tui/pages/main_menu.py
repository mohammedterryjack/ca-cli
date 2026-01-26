from textual.screen import Screen
from textual.widgets import Header, Footer, Button, Static
from textual.containers import Horizontal
from pyfiglet import Figlet
from tui.pages.ca import CAView
from tui.pages.ffnn import FFNNPredictView
from tui.pages.emergence import EmergenceView
from tui.pages.gpt import GPTGenerateView


class MainMenu(Screen):
    def compose(self):
        yield Header()

        f = Figlet(font="standard")  # you can try "smslant", "standard", etc.
        banner_text = f.renderText("Emergence Lab")
        yield Static(banner_text, classes="title")

        with Horizontal():
            yield Button("Cellular Automata", id="ca")
            yield Button("Emergence", id="emergence")
            yield Button("Feed-Forward Neural-Net", id="ffnn")
            yield Button("Character GPT", id="gpt")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "ca":
                self.app.push_screen(CAView())
            case "ffnn":
                self.app.push_screen(FFNNPredictView())
            case "emergence":
                self.app.push_screen(EmergenceView())
            case "gpt":
                self.app.push_screen(GPTGenerateView())
            case "exit":
                self.app.exit()
