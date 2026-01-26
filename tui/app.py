from textual.app import App
from tui.pages.main_menu import MainMenu


class EmergenceLabApp(App):
    CSS_PATH = "style.css"

    def on_mount(self):
        self.push_screen(MainMenu())


if __name__ == "__main__":
    EmergenceLabApp().run()
