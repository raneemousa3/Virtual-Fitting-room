from PIL import Image, ImageDraw, ImageFilter
import os

# Create a 32x32 image with a transparent background
img = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Colors
PURPLE = (147, 87, 255, 255)  # Modern purple
PINK = (255, 87, 147, 255)    # Pink accent
WHITE = (255, 255, 255, 255)
SHADOW = (0, 0, 0, 60)
GOLD = (255, 215, 0, 255)     # Gold for measuring tape
CYAN = (0, 255, 255, 255)     # Cyan for circuit connections

# Create gradient background
for y in range(32):
    # Gradient from purple to pink
    r = int(147 + (y / 32) * (255 - 147))
    g = int(87 + (y / 32) * (87 - 87))
    b = int(255 + (y / 32) * (147 - 255))
    color = (r, g, b, 255)
    draw.line([(0, y), (31, y)], fill=color)

# Add a subtle glow effect
glow = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
glow_draw = ImageDraw.Draw(glow)
glow_draw.ellipse([-2, -2, 33, 33], fill=(255, 255, 255, 30))
glow = glow.filter(ImageFilter.GaussianBlur(radius=2))
img = Image.alpha_composite(img, glow)

# Draw the stylized "R" with circuit-like connections
# Main vertical line of R
draw.line([(12, 6), (12, 22)], fill=CYAN, width=2)
# Top horizontal line
draw.line([(12, 6), (20, 6)], fill=CYAN, width=2)
# Middle horizontal line
draw.line([(12, 14), (20, 14)], fill=CYAN, width=2)
# Diagonal line
draw.line([(20, 6), (20, 14)], fill=CYAN, width=2)
draw.line([(20, 14), (24, 22)], fill=CYAN, width=2)

# Add circuit nodes (dots)
nodes = [(12, 6), (20, 6), (12, 14), (20, 14), (24, 22)]
for node in nodes:
    draw.ellipse([node[0]-2, node[1]-2, node[0]+2, node[1]+2], fill=WHITE)

# Add small connecting lines for circuit effect
draw.line([(14, 8), (16, 8)], fill=CYAN, width=1)
draw.line([(14, 12), (16, 12)], fill=CYAN, width=1)
draw.line([(14, 16), (16, 16)], fill=CYAN, width=1)
draw.line([(14, 20), (16, 20)], fill=CYAN, width=1)

# Add a small measuring tape at the bottom
draw.line([(2, 28), (10, 28)], fill=GOLD, width=2)
draw.line([(2, 28), (2, 30)], fill=GOLD, width=1)
draw.line([(6, 28), (6, 30)], fill=GOLD, width=1)
draw.line([(10, 28), (10, 30)], fill=GOLD, width=1)

# Add a small avatar silhouette
draw.ellipse([(22, 22), (26, 26)], fill=WHITE)  # Head
draw.line([(24, 26), (24, 30)], fill=WHITE, width=2)  # Body

# Save in multiple sizes for better quality
img_16 = img.resize((16, 16), Image.Resampling.LANCZOS)
img_24 = img.resize((24, 24), Image.Resampling.LANCZOS)
img_32 = img

# Create multi-size ICO file
img_32.save('favicon.ico', format='ICO', sizes=[(16, 16), (24, 24), (32, 32)]) 