import pygame

from gui.app import IsingApp


def main() -> None:
    pygame.init()
    try:
        app = IsingApp()
        app.run()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()