"""Split a 2x2 reference sheet in reading order: front, left, right, back."""


def split_sheet(image):
    width, height = image.size
    if width < 128 or height < 128:
        raise ValueError('ภาพรวม 4 มุมต้องมีขนาดอย่างน้อย 128×128 พิกเซล')
    mid_x, mid_y = width // 2, height // 2
    return {
        'front': image.crop((0, 0, mid_x, mid_y)),
        'left': image.crop((mid_x, 0, width, mid_y)),
        'right': image.crop((0, mid_y, mid_x, height)),
        'back': image.crop((mid_x, mid_y, width, height)),
    }
