import time
import pygame
import PygameXtras as px

pygame.init()
screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()

c1 = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}


def circle(xy, color, radius=3, width=1):
    color = c1[color] if isinstance(color, str) else color
    pygame.draw.circle(screen, color, xy, radius, width)


def line(xy1, xy2, color, width=1):
    color = c1[color] if isinstance(color, str) else color
    pygame.draw.line(screen, color, xy1, xy2, width)


def rect(r, color, width=1):
    color = c1[color] if isinstance(color, str) else color
    pygame.draw.rect(screen, color, r, width)


def vector(xy1, xy2, color, width=1, arrow_length=15):
    line(xy1, xy2, color, width)
    v = pygame.Vector2(xy1[0] - xy2[0], xy1[1] - xy2[1])
    v.scale_to_length(arrow_length)
    line(xy2, (xy2[0] + v.rotate(25).x, xy2[1] + v.rotate(25).y), color, width)
    line(xy2, (xy2[0] + v.rotate(-25).x, xy2[1] + v.rotate(-25).y), color, width)


center = (screen.get_width() // 2, screen.get_height() // 2)

label = px.Label(screen, "", 40, (10, 10), "topleft", tc=(255, 0, 0))

last_frame = time.time()
while True:
    current_frame = time.time()
    dt = current_frame - last_frame
    last_frame = current_frame

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            exit()

    label.update_text(f"FPS: {round(clock.get_fps(), 2)}")

    screen.fill((0, 0, 0))
    label.draw()

    pygame.display.flip()
    clock.tick()
