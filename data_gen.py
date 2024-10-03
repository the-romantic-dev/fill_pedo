import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import random


def rotate_image(image, max_angle=25):
    """
    Поворачивает изображение на случайный угол в пределах [-max_angle, max_angle],
    заполняя черные области белым фоном.
    """
    angle = random.uniform(-max_angle, max_angle)

    # Создаем белый фон большего размера, чтобы избежать появления черных областей
    width, height = image.size
    diagonal = int((width ** 2 + height ** 2) ** 0.5)  # Вычисляем диагональ для расширения фона
    white_bg = Image.new("RGB", (diagonal, diagonal), (255, 255, 255))

    # Вставляем исходное изображение в центр белого фона
    offset = ((diagonal - width) // 2, (diagonal - height) // 2)
    white_bg.paste(image, offset)

    # Поворачиваем изображение
    rotated_image = white_bg.rotate(angle, resample=Image.BICUBIC)

    # Обрезаем изображение обратно до исходного размера
    left = (diagonal - width) // 2
    top = (diagonal - height) // 2
    right = left + width
    bottom = top + height
    rotated_image = rotated_image.crop((left, top, right, bottom))

    return rotated_image


def shift_image(image, max_shift=5):
    """
    Сдвигает изображение на случайное значение по горизонтали и вертикали,
    пока буква остается полностью видимой.
    """
    width, height = image.size
    x_shift = random.randint(-max_shift, max_shift)
    y_shift = random.randint(-max_shift, max_shift)

    # Создаем пустое изображение
    shifted_image = Image.new("RGB", (width, height), (255, 255, 255))
    shifted_image.paste(image, (x_shift, y_shift))

    return shifted_image


def add_noise(image, noise_factor=0.1):
    """
    Добавляет случайный шум к изображению.
    """
    # Преобразуем изображение в массив
    image_array = np.array(image)

    # Генерируем шум
    noise = np.random.randn(*image_array.shape) * noise_factor * 255

    # Добавляем шум и обрезаем значения до диапазона [0, 255]
    noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)

    # Преобразуем обратно в изображение
    noisy_image = Image.fromarray(noisy_image_array)

    return noisy_image


def augment_image(image_path, noise_factor, max_shift, max_angle):
    """
    Выполняет все три операции с изображением.
    """
    image = Image.open(image_path).convert('RGB')  # Открываем изображение

    # Применяем поворот
    rotated_image = rotate_image(image, max_angle)

    # Применяем сдвиг
    shifted_image = shift_image(rotated_image, max_shift)

    # Добавляем шум
    noisy_image = add_noise(shifted_image, noise_factor)

    return noisy_image

reference_folder = Path('data/reference')
train_folder = Path('data/train')
validate_folder = Path('data/validate')
test_folder = Path('data/test')

train_per_reference = 15
validate_per_reference = 5
test_per_reference = 10
for letter in ['A', 'B', 'C', 'D']:
    letter_folder = Path(reference_folder, letter)
    for img in os.listdir(letter_folder):
        ref_path = Path(reference_folder, letter, img)

        train_save_folder = Path(train_folder, letter)
        if not train_save_folder.exists():
            train_save_folder.mkdir()
        for i in range(train_per_reference):
            augmented_image = augment_image(ref_path, noise_factor=0.3, max_shift=5, max_angle=40)
            augmented_image.save(Path(train_save_folder, f'{img.split(".")[0]}_{i}_strong.png'))

            augmented_image = augment_image(ref_path, noise_factor=0.2, max_shift=3, max_angle=15)
            augmented_image.save(Path(train_save_folder, f'{img.split(".")[0]}_{i}_middle.png'))

            augmented_image = augment_image(ref_path, noise_factor=0.0, max_shift=1, max_angle=5)
            augmented_image.save(Path(train_save_folder, f'{img.split(".")[0]}_{i}_easy.png'))

        validate_save_folder = Path(validate_folder, letter)
        if not validate_save_folder.exists():
            validate_save_folder.mkdir()
        for i in range(test_per_reference):
            augmented_image = augment_image(ref_path, noise_factor=0.4, max_shift=10, max_angle=35)
            augmented_image.save(Path(validate_save_folder, f'{img.split(".")[0]}_{i}_strong.png'))

            augmented_image = augment_image(ref_path, noise_factor=0.3, max_shift=7, max_angle=20)
            augmented_image.save(Path(validate_save_folder, f'{img.split(".")[0]}_{i}_middle.png'))

            augmented_image = augment_image(ref_path, noise_factor=0.2, max_shift=1, max_angle=2)
            augmented_image.save(Path(validate_save_folder, f'{img.split(".")[0]}_{i}_easy.png'))

        test_save_folder = Path(test_folder, letter)
        if not test_save_folder.exists():
            test_save_folder.mkdir()
        for i in range(validate_per_reference):
            augmented_image = augment_image(ref_path, noise_factor=0.3, max_shift=5, max_angle=25)
            augmented_image.save(Path(test_save_folder, f'{img.split(".")[0]}_{i}_strong.png'))

            augmented_image = augment_image(ref_path, noise_factor=0.2, max_shift=3, max_angle=15)
            augmented_image.save(Path(test_save_folder, f'{img.split(".")[0]}_{i}_middle.png'))

            augmented_image = augment_image(ref_path, noise_factor=0.1, max_shift=0, max_angle=2)
            augmented_image.save(Path(test_save_folder, f'{img.split(".")[0]}_{i}_easy.png'))